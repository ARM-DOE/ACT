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


def test_calculate_profile_fit_pbl():
    # Read the ceilometer data. Note the profile fit is applied to *uncorrected*
    #   backscatter, so correct_ceil is deliberately not called here.
    ds = act.io.arm.read_arm_netcdf(DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc'))
    ntime = ds.sizes['time']

    # Call the Retrieval
    ds = act.retrievals.pbl_lidar.calculate_profile_fit_pbl(
        ds, parm="backscatter", fit_min_height=100.0, fit_max_height=2500.0, time_average="30min"
    )

    # Both fields are returned on the native time axis, not the resampled one
    for var in ("pbl_profile_fit", "elevated_layer_fit"):
        assert ds[var].dims == ("time",)
        assert ds[var].size == ntime

    # The retrieval is computed on 30 min windows and mapped back onto native
    #   time, so a 6 hour slice must hold exactly 12 distinct values
    subset = ds.sel(time=slice("2019-01-01T09:00:00", "2019-01-01T15:00:00"))
    zi, zc = subset.pbl_profile_fit.values, subset.elevated_layer_fit.values
    assert len(np.unique(zi[np.isfinite(zi)])) == 12
    assert len(np.unique(zc[np.isfinite(zc)])) == 12

    # Test the mean/min/max mixed-layer height over the subset
    np.testing.assert_array_almost_equal(np.nanmean(zi), 817.454, decimal=2)
    np.testing.assert_almost_equal(np.nanmin(zi), 739.759, decimal=2)
    np.testing.assert_almost_equal(np.nanmax(zi), 894.357, decimal=2)

    # Test the mean elevated aerosol layer height over the subset
    np.testing.assert_array_almost_equal(np.nanmean(zc), 1686.357, decimal=2)

    # The elevated layer must sit above the mixed layer wherever both converged,
    #   and both must stay inside the requested fit window
    full_zi = ds.pbl_profile_fit.values
    full_zc = ds.elevated_layer_fit.values
    both = np.isfinite(full_zi) & np.isfinite(full_zc)
    assert both.any()
    assert np.all(full_zc[both] > full_zi[both])
    assert np.all((full_zi[both] >= 100.0) & (full_zc[both] <= 2500.0))

    # test attributes
    for var in ("pbl_profile_fit", "elevated_layer_fit"):
        assert ds[var].attrs["input_parameter"] == "backscatter"
        assert ds[var].attrs["units"] == "m"
        assert ds[var].attrs["time_average"] == "30min"
        assert "Steyn, Baldi & Hoff (1999)" in ds[var].attrs["description"]

    # A coarser averaging window must produce correspondingly fewer retrievals
    ds_hr = act.io.arm.read_arm_netcdf(DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc'))
    ds_hr = act.retrievals.pbl_lidar.calculate_profile_fit_pbl(
        ds_hr, parm="backscatter", time_average="60min"
    )
    hourly = ds_hr.pbl_profile_fit.values
    assert ds_hr.pbl_profile_fit.size == ntime
    assert len(np.unique(hourly[np.isfinite(hourly)])) == 23
    assert ds_hr['pbl_profile_fit'].attrs["time_average"] == "60min"

    # Disabling the elevated layer must still run and must return no elevated heights
    ds_single = act.io.arm.read_arm_netcdf(DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc'))
    ds_single = act.retrievals.pbl_lidar.calculate_profile_fit_pbl(
        ds_single, parm="backscatter", allow_elevated=False
    )
    assert ds_single.pbl_profile_fit.size == ntime
    assert np.all(np.isnan(ds_single.elevated_layer_fit.values))


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
