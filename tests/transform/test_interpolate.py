"""Behavior tests for act.transform.interpolate."""

import numpy as np
import pytest
import xarray as xr
from _helpers import MISSING, _da

import act


def test_basic_1d():
    da = _da([0.0, 1.0, 4.0, 9.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
    target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=["time"])
    result, qc = act.transform.interpolate(da, target, dim="time")
    assert result.shape == (3,)
    assert result.values[0] == pytest.approx(0.5)
    assert result.values[1] == pytest.approx(2.5)
    assert result.name == "temp"


def test_2d_dataarray():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    height = np.array([100.0, 200.0, 300.0])
    data = np.arange(12, dtype=float).reshape(4, 3)
    da = xr.DataArray(
        data,
        coords={"time": time, "height": height},
        dims=["time", "height"],
        name="wind",
    )
    target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=["time"])
    result, qc = act.transform.interpolate(da, target, dim="time")
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result.values, (data[:-1] + data[1:]) / 2)
    np.testing.assert_array_equal(qc.values, np.zeros((3, 3)))


def test_missing_value_respected():
    da = _da([0.0, MISSING, 4.0], coord=np.array([0.0, 1.0, 2.0]))
    da.encoding["_FillValue"] = MISSING
    target = np.array([0.5, 1.5])
    result, qc = act.transform.interpolate(da, target, dim="time")
    assert qc.values[1] != 0


def test_default_t_range_is_unlimited():
    # libtrans places no range limit by default, so the missing neighbour
    # is stepped over rather than failing the target.
    da = _da([0.0, 10.0, MISSING, 30.0, 40.0])
    da.encoding["_FillValue"] = MISSING
    result, qc = act.transform.interpolate(da, np.array([2.5]), dim="time")
    assert result.values[0] == pytest.approx(25.0)
    assert qc.values[0] & act.transform.QC_INTERPOLATE
    limited, limited_qc = act.transform.interpolate(da, np.array([2.5]), dim="time", t_range=1.0)
    assert limited_qc.values[0] & act.transform.QC_OUTSIDE_RANGE


def test_mismatched_ordering_raises():
    da = _da([0.0, 1.0, 2.0, 3.0])
    target = xr.DataArray(np.array([2.5, 0.5]), dims=["time"])
    with pytest.raises(ValueError, match="status -5"):
        act.transform.interpolate(da, target, dim="time")


@pytest.mark.parametrize(
    "values, flags, target, t_range, expected, expected_qc",
    [
        ([0, 100, 20], [0, 1, 0], 0.5, None, 5.0, act.transform.QC_INTERPOLATE),
        ([0, 10, 20], [2, 0, 0], 0.5, None, 5.0, act.transform.QC_INDETERMINATE),
        (
            [0, 10, 20],
            [0, 0, 0],
            0.5,
            0.4,
            MISSING,
            act.transform.QC_OUTSIDE_RANGE | act.transform.QC_BAD,
        ),
    ],
)
def test_qc_and_range(values, flags, target, t_range, expected, expected_qc):
    da = _da(values)
    qc = xr.DataArray(flags, coords=da.coords, dims=da.dims)
    result, result_qc = act.transform.interpolate(
        da, [target], dim="time", qc=qc, qc_mask=1, t_range=t_range
    )
    assert result.values[0] == pytest.approx(expected)
    assert result_qc.values[0] == expected_qc


def test_single_input_is_outside_range():
    result, qc = act.transform.interpolate(_da([10]), [0], dim="time")
    np.testing.assert_array_equal(result.values, [MISSING])
    np.testing.assert_array_equal(
        qc.values, [act.transform.QC_BAD | act.transform.QC_OUTSIDE_RANGE]
    )


def test_only_one_usable_input_marks_all_targets_bad():
    da = _da([10, MISSING, MISSING])
    result, qc = act.transform.interpolate(da, [0.5, 1.5], dim="time")
    np.testing.assert_array_equal(result.values, [MISSING, MISSING])
    assert np.all(qc.values & act.transform.QC_ALL_BAD_INPUTS)
    assert np.all(qc.values & act.transform.QC_BAD)


def test_matched_descending_order():
    result, qc = act.transform.interpolate(
        _da([20, 10, 0], coord=[2, 1, 0]), [1.5, 0.5], dim="time"
    )
    np.testing.assert_allclose(result.values, [15, 5])
    np.testing.assert_array_equal(qc.values, [0, 0])
