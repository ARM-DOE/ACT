import numpy as np
import xarray as xr

import act


def test_correct_ceil():
    # Make a fake ARM dataset to test with, just an array with 1e-7 for half
    # of it
    fake_data = 10 * np.ones((300, 20))
    fake_data[:, 10:] = -1
    arm_ds = {}
    arm_ds['backscatter'] = xr.DataArray(fake_data)
    arm_ds = act.corrections.ceil.correct_ceil(arm_ds)
    assert np.all(arm_ds['backscatter'].data[:, 10:] == -7)
    assert np.all(arm_ds['backscatter'].data[:, 1:10] == 1)

    arm_ds['backscatter'].attrs['units'] = 'dummy'
    arm_ds = act.corrections.ceil.correct_ceil(arm_ds)
    assert arm_ds['backscatter'].units == 'log(dummy)'
