import numpy as np
from arm_test_data import DATASETS

import act


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
