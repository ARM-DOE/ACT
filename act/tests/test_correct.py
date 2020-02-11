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

    assert np.all(np.round(obj['cross_co_ratio'].data[0, 500]) == 51.)
    assert np.all(np.round(obj['signal_return_co_pol'].data[0, 11]) == 11)
    assert np.all(np.round(obj['signal_return_co_pol'].data[0, 500]) == -6)


def test_correct_wind():
    nav = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_NAV)
    nav = act.utils.ship_utils.calc_cog_sog(nav)

    aosmet = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_AOSMET)

    obj = xr.merge([nav, aosmet], compat='override')
    obj = act.corrections.ship.correct_wind(obj)

    assert round(obj['wind_speed_corrected'].values[800]) == 5.0
    assert round(obj['wind_direction_corrected'].values[800]) == 92.0


def test_correct_dl():
    # Test the DL correction script on a PPI dataset eventhough it will
    # mostlikely be used on FPT scans. Doing this to save space with only
    # one datafile in the repo.
    files = act.tests.sample_files.EXAMPLE_DLPPI
    obj = act.io.armfiles.read_netcdf(files)

    new_obj = act.corrections.doppler_lidar.correct_dl(obj, fill_value=np.nan)
    data = new_obj['attenuated_backscatter'].data
    data[np.isnan(data)] = 0.
    data = data * 100.
    data = data.astype(np.int64)
    assert np.sum(data) == -18633551

    new_obj = act.corrections.doppler_lidar.correct_dl(obj, range_normalize=False)
    data = new_obj['attenuated_backscatter'].data
    data[np.isnan(data)] = 0.
    data = data.astype(np.int64)
    assert np.sum(data) == -224000
