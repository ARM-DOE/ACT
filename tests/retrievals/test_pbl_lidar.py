import numpy as np
import pytest
from arm_test_data import DATASETS

import act

try:
    import pywt  # noqa

    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False


def test_calculate_gradient_pbl():
    # Read and apply connections
    ds = act.io.arm.read_arm_netcdf(DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc'))
    ds = act.corrections.correct_ceil(ds, var_name='backscatter')

    # Call the Retrieval
    ds = act.retrievals.pbl_lidar.calculate_gradient_pbl(
        ds, parm="backscatter", smooth_dis=3, min_height=200
    )

    # create a subset for testing
    subset = ds.sel(time=slice("2019-01-01T11:30:00", "2019-01-01T11:40:00"))
    # Test the mean of the profile for the subset time
    np.testing.assert_array_almost_equal(subset.pbl_gradient.mean(), 436.875, decimal=3)
    # Test the minimum PBL Height during the period
    #   note this will test the minimum height threshold assigned
    np.testing.assert_almost_equal(subset.pbl_gradient.min(), 225.0, 1)

    # test attributes
    assert ds['pbl_gradient'].attrs["input_parameter"] == "backscatter"
    assert ds['pbl_gradient'].attrs["units"] == "m"


def test_calculate_modified_gradient_pbl():
    # Read and apply connections
    ds = act.io.arm.read_arm_netcdf(DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc'))
    ds = act.corrections.correct_ceil(ds, var_name='backscatter')

    # Call the Retrieval
    ds = act.retrievals.pbl_lidar.calculate_modified_gradient_pbl(
        ds, parm="backscatter", smooth_dis=3, min_height=200, threshold=1e-5
    )

    # create a subset for testing
    subset = ds.sel(time=slice("2019-01-01T11:30:00", "2019-01-01T11:40:00"))
    # Test the mean of the profile for the subset time
    np.testing.assert_array_almost_equal(subset.pbl_mod_gradient.mean(), 372.631, decimal=3)
    # Test the minimum PBL Height during the period
    #   note this will test the minimum height threshold assigned
    np.testing.assert_almost_equal(subset.pbl_mod_gradient.min(), 225.0, 1)

    # test attributes
    assert ds['pbl_mod_gradient'].attrs["input_parameter"] == "backscatter"
    assert ds['pbl_mod_gradient'].attrs["units"] == "m"


@pytest.mark.skipif(not PYWAVELETS_AVAILABLE, reason='PyWavelets is not installed.')
def test_calculate_wavelet_pbl():
    # Read and apply connections
    ds = act.io.arm.read_arm_netcdf(DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc'))
    ds = act.corrections.correct_ceil(ds, var_name='backscatter')

    # Call the Retrieval
    ds = act.retrievals.pbl_lidar.calculate_wavelet_pbl(
        ds, var_name='backscatter', range_name='range', scale=60.0
    )

    # create a subset for testing
    subset = ds.sel(resampled_time=slice("2019-01-01T11:30:00", "2019-01-01T11:40:00"))
    # Test the mean of the profile for the subset time
    np.testing.assert_array_almost_equal(subset.pbl_wavelet.mean(), 4569.961, decimal=3)
    # Test the minimum PBL Height during the period
    np.testing.assert_almost_equal(subset.pbl_wavelet.min(), 3435.0, 1)

    # test attributes
    assert ds['pbl_wavelet'].attrs["input_parameter"] == "backscatter"
    assert ds['pbl_wavelet'].attrs["units"] == "m"


@pytest.mark.skipif(not PYWAVELETS_AVAILABLE, reason='PyWavelets is not installed.')
def test_calculate_wavelet_pbl_max_height():
    # Read and apply connections
    ds = act.io.arm.read_arm_netcdf(DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc'))
    ds = act.corrections.correct_ceil(ds, var_name='backscatter')

    # Call the Retrieval, constraining the search to below 1000 m
    ds = act.retrievals.pbl_lidar.calculate_wavelet_pbl(
        ds, var_name='backscatter', range_name='range', scale=60.0, max_height=1000.0
    )

    # create a subset for testing
    subset = ds.sel(resampled_time=slice("2019-01-01T11:30:00", "2019-01-01T11:40:00"))
    # Test that the upper bound is respected and no longer picks up the
    # higher-altitude layers found without max_height set
    assert subset.pbl_wavelet.max() <= 1000.0
    np.testing.assert_array_almost_equal(subset.pbl_wavelet.mean(), 615.0, decimal=3)
