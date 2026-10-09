"""Behavior tests for act.transform.bin_average."""

import numpy as np
import pytest
import xarray as xr
from _helpers import MISSING, _da

import act


def test_basic_1d():
    da = _da([0.0, 2.0, 4.0, 6.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
    target = xr.DataArray(np.array([1.0, 3.0]), dims=["time"])
    result, qc = act.transform.bin_average(da, target, dim="time")
    assert result.shape == (2,)
    np.testing.assert_allclose(result.values, [2.0, 8.0 / 1.5])
    np.testing.assert_array_equal(qc.values, [0, 0])


def test_mismatched_ordering_raises():
    da = _da([0.0, 2.0, 4.0, 6.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
    target = xr.DataArray(np.array([3.0, 1.0]), dims=["time"])
    with pytest.raises(ValueError, match="status -5"):
        act.transform.bin_average(da, target, dim="time")


def test_matched_descending_ordering_succeeds():
    da = _da([6.0, 4.0, 2.0, 0.0], coord=np.array([3.0, 2.0, 1.0, 0.0]))
    target = xr.DataArray(np.array([2.5, 0.5]), dims=["time"])
    result, qc = act.transform.bin_average(da, target, dim="time")
    np.testing.assert_allclose(result.values, [5.0, 1.0])


def test_qc_mask_assessment_and_default():
    da = _da([1.0, 2.0], coord=np.array([0.0, 1.0]))
    qc = xr.DataArray([1, 2], coords=da.coords, dims=da.dims)
    qc.attrs["flag_masks"] = [1, 2]
    qc.attrs["flag_assessments"] = ["Bad", "Indeterminate"]
    target = xr.DataArray([0.0], dims=["time"])
    bounds = np.array([[0.0, 2.0]])
    input_bounds = np.array([[0.0, 1.0], [1.0, 2.0]])

    _, default_qc = act.transform.bin_average(
        da,
        target,
        dim="time",
        qc=qc,
        input_bounds=input_bounds,
        output_bounds=bounds,
    )
    _, bad_qc = act.transform.bin_average(
        da,
        target,
        dim="time",
        qc=qc,
        qc_mask="Bad",
        input_bounds=input_bounds,
        output_bounds=bounds,
    )
    _, indeterminate_qc = act.transform.bin_average(
        da,
        target,
        dim="time",
        qc=qc,
        qc_mask="Indeterminate",
        input_bounds=input_bounds,
        output_bounds=bounds,
    )

    np.testing.assert_array_equal(default_qc.values, bad_qc.values)
    assert bad_qc.values[0] & act.transform.QC_SOME_BAD_INPUTS
    assert indeterminate_qc.values[0] & act.transform.QC_SOME_BAD_INPUTS


def test_qc_matching_or_transposed_dimensions_produce_identical_results():
    time = np.arange(6, dtype=float)
    height = [100, 200]
    da = xr.DataArray(
        np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]]),
        dims=("time", "height"),
        coords={"time": time, "height": height},
    )
    qc = xr.DataArray(
        np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
        dims=da.dims,
        coords=da.coords,
    )
    target = xr.DataArray([1.0, 3.0, 5.0], dims=["time"])

    expected_data, expected_qc = act.transform.bin_average(da, target, dim="time", qc=qc, qc_mask=1)
    actual_data, actual_qc = act.transform.bin_average(
        da, target, dim="time", qc=qc.transpose("height", "time"), qc_mask=1
    )

    np.testing.assert_array_equal(expected_data.values[:, 1], MISSING)
    xr.testing.assert_identical(actual_data, expected_data)
    xr.testing.assert_identical(actual_qc, expected_qc)


@pytest.mark.parametrize(
    "change",
    [
        lambda qc: qc.rename(height="level"),
        lambda qc: qc.assign_coords(time=np.arange(6, dtype=float) + 1),
        lambda qc: qc.assign_coords(height=[200, 100]),
        lambda qc: qc.isel(time=slice(None, -1)),
        lambda qc: qc.drop_indexes("height").drop_vars("height"),
    ],
    ids=[
        "wrong-dimensions",
        "wrong-time",
        "wrong-height",
        "wrong-shape",
        "missing-coordinate",
    ],
)
def test_qc_mismatched_dimensions_or_coordinates_raise(change):
    da = xr.DataArray(
        np.arange(12).reshape(6, 2),
        dims=("time", "height"),
        coords={"time": np.arange(6, dtype=float), "height": [100, 200]},
    )
    qc = xr.zeros_like(da, dtype=np.int32)

    with pytest.raises(ValueError, match="QC.*(dimensions|coordinates|shape)"):
        act.transform.bin_average(da, [1.0, 3.0, 5.0], dim="time", qc=change(qc), qc_mask=1)


def test_qc_without_flag_masks_requires_explicit_mask():
    da = _da([1.0, 100.0, 1.0, 1.0])
    qc = xr.DataArray([0, 1, 0, 0], coords=da.coords, dims=da.dims)
    target = np.array([0.5, 2.5])
    bounds = np.array([[0.0, 2.0], [2.0, 4.0]])
    with pytest.raises(ValueError, match="flag_masks"):
        act.transform.bin_average(da, target, dim="time", qc=qc, output_bounds=bounds)
    result, _ = act.transform.bin_average(
        da,
        target,
        dim="time",
        qc=qc,
        qc_mask=1,
        input_bounds=np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        output_bounds=bounds,
    )
    np.testing.assert_allclose(result.values, [1.0, 1.0])


@pytest.mark.parametrize(
    "bounds, flags, kwargs, expected, expected_qc",
    [
        ([[0, 2]], [0, 0], {"weights": [1, 3]}, 7.5, 0),
        ([[0, 1.25]], [0, 0], {}, 2.0, 0),
        (
            [[0, 2]],
            [0, 1],
            {"goodfrac_bad_min": 0.75},
            0,
            act.transform.QC_SOME_BAD_INPUTS | act.transform.QC_BAD_GOODFRAC,
        ),
        (
            [[0, 2]],
            [0, 0],
            {"std_ind_max": 4},
            5,
            act.transform.QC_INDETERMINATE_STD,
        ),
        ([[0, 2]], [0, 0], {"weights": [0, 0]}, 0, act.transform.QC_ZERO_WEIGHT),
        (
            [[3, 4]],
            [0, 0],
            {},
            MISSING,
            act.transform.QC_OUTSIDE_RANGE | act.transform.QC_BAD,
        ),
        ([[0, 2]], [0, 0], {"std_bad_max": 4}, 5, act.transform.QC_BAD_STD),
        (
            [[0, 2]],
            [0, 1],
            {"goodfrac_ind_min": 0.75},
            0,
            act.transform.QC_SOME_BAD_INPUTS | act.transform.QC_INDETERMINATE_GOODFRAC,
        ),
    ],
)
def test_weights_overlap_and_thresholds(bounds, flags, kwargs, expected, expected_qc):
    da = _da([0, 10], coord=[0.5, 1.5])
    qc = xr.DataArray(flags, coords=da.coords, dims=da.dims)
    result, result_qc = act.transform.bin_average(
        da,
        [1.0],
        dim="time",
        qc=qc,
        qc_mask=1,
        input_bounds=[[0, 1], [1, 2]],
        output_bounds=bounds,
        **kwargs,
    )
    assert result.values[0] == pytest.approx(expected)
    assert result_qc.values[0] == expected_qc


def test_inconsistent_input_bounds_raise():
    with pytest.raises(ValueError, match="status -1"):
        act.transform.bin_average(
            _da([10, 20], coord=[0.5, 1.5]),
            [1.0],
            dim="time",
            input_bounds=[[0, -1], [1, 2]],
            output_bounds=[[-2, 2]],
        )


def test_wrong_length_weights_raise():
    da = _da([0.0, 2.0, 4.0, 6.0])
    target = np.array([1.0, 3.0])
    with pytest.raises(ValueError, match="weights"):
        act.transform.bin_average(da, target, dim="time", weights=np.ones(2))


def test_zero_width_output_bins_raise():
    da = _da([0.0, 2.0, 4.0, 6.0])
    with pytest.raises(ValueError, match="nonzero width"):
        act.transform.bin_average(da, np.array([1.5]), dim="time")
    with pytest.raises(ValueError, match="nonzero width"):
        act.transform.bin_average(
            da,
            np.array([0.5, 2.5]),
            dim="time",
            output_bounds=np.array([[0.0, 1.0], [2.5, 2.5]]),
        )
    result, _ = act.transform.bin_average(
        da, np.array([1.5]), dim="time", output_bounds=np.array([[-0.5, 3.5]])
    )
    assert result.values[0] == pytest.approx(3.0)


def test_multiple_qc_assessments_are_combined():
    da = _da([1.0, 2.0, 3.0], coord=np.array([0.0, 1.0, 2.0]))
    qc = xr.DataArray([1, 2, 4], coords=da.coords, dims=da.dims)
    qc.attrs["flag_masks"] = [1, 2, 4]
    qc.attrs["flag_assessments"] = ["Bad", "Suspect", "Indeterminate"]
    target = xr.DataArray([1.0], dims=["time"])
    result, _ = act.transform.bin_average(
        da,
        target,
        dim="time",
        qc=qc,
        qc_mask=["Bad", "Suspect"],
        input_bounds=np.array([[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]]),
        output_bounds=np.array([[0.0, 2.0]]),
    )
    assert result.values[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "mask, exception, message",
    [
        ("Suspect", ValueError, "not found"),
        (["Bad", 1], TypeError, "assessment string"),
    ],
)
def test_invalid_qc_assessment_raises(mask, exception, message):
    da = _da([1.0, 2.0])
    qc = xr.DataArray([0, 0], coords=da.coords, dims=da.dims)
    qc.attrs.update(flag_masks=[1], flag_assessments=["Bad"])
    with pytest.raises(exception, match=message):
        act.transform.bin_average(da, [0.5], dim="time", qc=qc, qc_mask=mask)


@pytest.mark.parametrize(
    "flags, expected_qc",
    [
        ([act.transform.QC_BAD, 0], act.transform.QC_SOME_BAD_INPUTS),
        (
            [act.transform.QC_BAD] * 2,
            act.transform.QC_ALL_BAD_INPUTS | act.transform.QC_BAD,
        ),
    ],
)
def test_partial_vs_all_bad_inputs(flags, expected_qc):
    da = _da([1.0, 2.0])
    qc = xr.DataArray(flags, coords=da.coords, dims=da.dims)
    _, result_qc = act.transform.bin_average(
        da,
        [1.0],
        dim="time",
        qc=qc,
        qc_mask=act.transform.QC_BAD,
        input_bounds=[[0.0, 1.0], [1.0, 2.0]],
        output_bounds=[[0.0, 2.0]],
    )
    assert result_qc.values[0] == expected_qc
