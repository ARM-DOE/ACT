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


def test_correct_mpl():
    # Make a fake ARM dataset to test with, just an array with 1e-7 for half
    # of it
    files = act.tests.sample_files.EXAMPLE_MPL_1SAMPLE
    obj = act.io.armfiles.read_netcdf(files)

    obj = act.corrections.mpl.correct_mpl(obj)

    assert np.all(np.round(obj['signal_return_co_pol'].data[0, 11]) == 11)
    assert np.all(np.round(obj['signal_return_co_pol'].data[0, 500]) == -6)
