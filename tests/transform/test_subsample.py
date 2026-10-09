"""Behavior tests for act.transform.subsample."""

import numpy as np
import pytest
import xarray as xr
from _helpers import MISSING, _da

import act


def test_basic_1d():
    da = _da([10.0, 20.0, 30.0], coord=np.array([0.0, 1.0, 2.0]))
    target = np.array([0.1, 0.9, 1.9])
    result, qc = act.transform.subsample(da, target, dim="time", t_range=0.5)
    assert result.shape == (3,)
    assert result.values[0] == pytest.approx(10.0)
    assert result.values[1] == pytest.approx(20.0)
    assert result.values[2] == pytest.approx(30.0)


@pytest.mark.parametrize(
    "values, flags, target, expected, expected_qc",
    [
        ([10, 20, 30], [0, 1, 0], 0.9, 10, act.transform.QC_NOT_USING_CLOSEST),
        ([10, 20, 30], [2, 0, 0], 0.1, 10, act.transform.QC_INDETERMINATE),
        (
            [10, 20, 30],
            [1, 1, 1],
            0.9,
            MISSING,
            act.transform.QC_ALL_BAD_INPUTS | act.transform.QC_BAD,
        ),
    ],
)
def test_qc_selection(values, flags, target, expected, expected_qc):
    da = _da(values)
    qc = xr.DataArray(flags, coords=da.coords, dims=da.dims)
    result, result_qc = act.transform.subsample(
        da, [target], dim="time", qc=qc, qc_mask=1, t_range=1.0
    )
    assert result.values[0] == pytest.approx(expected)
    assert result_qc.values[0] == expected_qc


def test_bad_tail_distinguishes_all_bad_from_outside_range():
    da = _da([10, 20, 30], coord=[0, 1, 2])
    qc = xr.DataArray([0, 0, 1], coords=da.coords, dims=da.dims)
    result, result_qc = act.transform.subsample(
        da, [1.9, 2.1, 4.0], dim="time", qc=qc, qc_mask=1, t_range=0.5
    )
    np.testing.assert_array_equal(result.values, [MISSING] * 3)
    np.testing.assert_array_equal(
        result_qc.values,
        [
            act.transform.QC_ALL_BAD_INPUTS | act.transform.QC_BAD,
            act.transform.QC_ALL_BAD_INPUTS | act.transform.QC_BAD,
            act.transform.QC_OUTSIDE_RANGE | act.transform.QC_BAD,
        ],
    )


def test_targets_outside_range_do_not_raise():
    da = _da([10.0, 20.0, 30.0], coord=np.array([0.0, 1.0, 2.0]))
    target = np.array([0.9, 5.0])
    result, qc = act.transform.subsample(da, target, dim="time", t_range=0.5)
    assert result.values[0] == pytest.approx(20.0)
    assert qc.values[1] & act.transform.constants.QC_OUTSIDE_RANGE
