import act
import numpy as np
import xarray as xr


def test_correct_ceil():
    # Make a fake ARM dataset to test with, just an array with 1e-7 for half
    # of it
    fake_data = 10 * np.ones((300, 20))
    fake_data[:, 10:] = -1
    arm_obj = {}
    arm_obj['backscatter'] = xr.DataArray(fake_data)
    arm_obj = act.corrections.ceil.correct_ceil(arm_obj)
    assert np.all(arm_obj['backscatter'].data[:, 10:] == -7)
    assert np.all(arm_obj['backscatter'].data[:, 1:10] == 1)
