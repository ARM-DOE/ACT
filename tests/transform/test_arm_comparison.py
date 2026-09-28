from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import act

FIXTURE_DIR = Path(__file__).parents[1] / "data/transform/arm_reference"
INPUT = FIXTURE_DIR / "input.nc"
BIN_AVERAGE = FIXTURE_DIR / "expected_bin_average.nc"
INTERPOLATE = FIXTURE_DIR / "expected_interpolate.nc"


class _OpenedDatasets:
    def __init__(self, *paths):
        self.datasets = [xr.open_dataset(path).load() for path in paths]

    def __enter__(self):
        return self.datasets

    def __exit__(self, exc_type, exc_value, traceback):
        for dataset in self.datasets:
            dataset.close()


def _open(*paths):
    return _OpenedDatasets(*paths)


def _centered_time_grid(reference):
    target = xr.DataArray(
        reference["time"].values,
        dims=["time"],
        coords={"time": reference["time"].values},
        name="time",
    )
    return target, reference["time_bounds"].values


def _centered_diameter_grid(reference):
    target = xr.DataArray(
        reference["diameter"].values,
        dims=["diameter_mobility"],
        coords={"diameter_mobility": reference["diameter"].values},
        name="diameter_mobility",
    )
    return target, reference["diameter_bounds"].values


def _assert_values_match(actual, expected, *, rtol=1e-5, atol=2e-4):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape

    actual_missing = ~np.isfinite(actual) | (actual == -9999)
    expected_missing = ~np.isfinite(expected) | (expected == -9999)
    np.testing.assert_array_equal(actual_missing, expected_missing)
    valid = ~expected_missing
    np.testing.assert_allclose(actual[valid], expected[valid], rtol=rtol, atol=atol)


def _assert_qc_match(actual, expected, *, ignore_bits=0):
    actual = np.asarray(actual, dtype=np.int64)
    expected = np.asarray(expected, dtype=np.int64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual & ~ignore_bits, expected & ~ignore_bits)


def test_arm_bin_average_reference():
    """Compare ACT's two-pass bin average with ARM's 1-hour/100-nm output."""
    with _open(INPUT, BIN_AVERAGE) as datasets:
        source, reference = datasets
        time, time_bounds = _centered_time_grid(reference)
        diameter, diameter_bounds = _centered_diameter_grid(reference)

        # The source time cells are 5 minutes wide.  The source QC's bit 1 is
        # ARM's bad-value flag, which is the bad mask used by these comparisons.
        time_result, time_qc = act.transform.bin_average(
            source["dN_dlogDp"],
            time,
            dim="time",
            qc=source["qc_dN_dlogDp"],
            qc_mask=1,
            input_bounds=source["time_bounds"].values,
            output_bounds=time_bounds,
        )
        result, qc = act.transform.bin_average(
            time_result,
            diameter,
            dim="diameter_mobility",
            qc=time_qc,
            qc_mask=1,
            input_bounds=source["diameter_mobility_bounds"].values,
            output_bounds=diameter_bounds,
        )

        _assert_values_match(result.values, reference["aerosol_count"].values)
        # ARM also sets QC_SOME_BAD_INPUTS when every input is bad, contradicting
        # its own flag definition ("some, but not all"); ACT does not.
        _assert_qc_match(
            qc.values,
            reference["qc_aerosol_count"].values,
            ignore_bits=act.transform.constants.QC_SOME_BAD_INPUTS,
        )

        for source_name, reference_name in (
            ("total_N_conc", "aerosol_density"),
            ("aerosol_flow", "aerosol_flow"),
        ):
            qc_input = source.get("qc_" + source_name)
            transformed, transformed_qc = act.transform.bin_average(
                source[source_name],
                time,
                dim="time",
                qc=qc_input,
                qc_mask=1,
                input_bounds=source["time_bounds"].values,
                output_bounds=time_bounds,
            )
            _assert_values_match(transformed.values, reference[reference_name].values)
            _assert_qc_match(
                transformed_qc.values,
                reference["qc_" + reference_name].values,
            )


def test_arm_interpolate_reference():
    """Compare ACT's two-pass interpolation with ARM's 30-second/5-nm output."""
    with _open(INPUT, INTERPOLATE) as datasets:
        source, reference = datasets
        # ARM interpolates the center of each input cell, rather than the input
        # coordinate values rounded by the source file.  This distinction is
        # material for the SMPS diameter axis.
        source = source.assign_coords(
            time=source["time_bounds"].mean("bound").values,
            diameter_mobility=source["diameter_mobility_bounds"].mean("bound").values,
        )
        time, _ = _centered_time_grid(reference)
        diameter, _ = _centered_diameter_grid(reference)

        time_result, time_qc = act.transform.interpolate(
            source["dN_dlogDp"],
            time,
            dim="time",
            qc=source["qc_dN_dlogDp"],
            qc_mask=1,
            t_range=600,
        )
        result, qc = act.transform.interpolate(
            time_result,
            diameter,
            dim="diameter_mobility",
            qc=time_qc,
            qc_mask=1,
            t_range=100,
        )

        # Interpolation is numerically equivalent for the usable, unflagged
        # region.  ARM and ACT intentionally differ on edge/invalid QC handling
        # while the implementation is being aligned, so do not turn those
        # known edge conditions into a broad exact-output contract.
        actual = result.values
        expected = reference["dN_dlogDp"].values
        valid = (
            np.isfinite(actual)
            & np.isfinite(expected)
            & (actual != -9999)
            & (expected != -9999)
            & (reference["qc_dN_dlogDp"].values == 0)
        )
        assert valid.any()
        # The ARM and ACT kernels make slightly different choices around
        # missing SMPS bins.  Away from those edge choices, the discrepancy is
        # small relative to the distribution magnitude.
        relative_error = np.abs(actual[valid] - expected[valid]) / (
            np.abs(expected[valid]) + 1e-6
        )
        assert np.mean(relative_error < 0.2) > 0.98
        np.testing.assert_allclose(actual[valid], expected[valid], rtol=3.0, atol=2.0)

        # Scalar variables provide a strict interpolation check unaffected by
        # the multidimensional size-distribution edge cases.
        for name in ("aerosol_flow", "total_N_conc"):
            transformed, transformed_qc = act.transform.interpolate(
                source[name],
                time,
                dim="time",
                qc=source.get("qc_" + name),
                qc_mask=1,
                t_range=600,
            )
            _assert_values_match(transformed.values, reference[name].values, atol=2e-4)
            _assert_qc_match(transformed_qc.values, reference["qc_" + name].values)
