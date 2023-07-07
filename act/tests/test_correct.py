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


def test_correct_mpl():
    # Make a fake ARM dataset to test with, just an array with 1e-7 for half
    # of it
    test_data = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MPL_1SAMPLE)
    ds = act.corrections.mpl.correct_mpl(test_data)
    sig_cross_pol = ds['signal_return_cross_pol'].values[1, 10:15]
    sig_co_pol = ds['signal_return_co_pol'].values[1, 10:15]
    height = ds['height'].values[0:10]
    overlap0 = ds['overlap_correction'].values[1, 0, 0:5]
    overlap1 = ds['overlap_correction'].values[1, 1, 0:5]
    overlap2 = ds['overlap_correction'].values[1, 2, 0:5]
    np.testing.assert_allclose(overlap0, [0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(overlap1, [754.338, 754.338, 754.338, 754.338, 754.338])
    np.testing.assert_allclose(overlap2, [181.9355, 181.9355, 181.9355, 181.9355, 181.9355])
    np.testing.assert_allclose(
        sig_cross_pol,
        [-0.5823283, -1.6066532, -1.7153032, -2.520143, -2.275405],
        rtol=4e-06,
    )
    np.testing.assert_allclose(
        sig_co_pol, [12.5631485, 11.035495, 11.999875, 11.09393, 11.388968], rtol=1e-6
    )
    np.testing.assert_allclose(
        height,
        [
            0.00749012,
            0.02247084,
            0.03745109,
            0.05243181,
            0.06741206,
            0.08239277,
            0.09737302,
            0.11235374,
            0.12733398,
            0.14231472,
        ],
        rtol=1e-6,
    )
    assert ds['signal_return_co_pol'].attrs['units'] == '10 * log10(count/us)'
    assert ds['signal_return_cross_pol'].attrs['units'] == '10 * log10(count/us)'
    assert ds['cross_co_ratio'].attrs['long_name'] == 'Cross-pol / Co-pol ratio * 100'
    assert ds['cross_co_ratio'].attrs['units'] == '1'
    assert 'description' not in ds['cross_co_ratio'].attrs.keys()
    assert 'ancillary_variables' not in ds['cross_co_ratio'].attrs.keys()
    assert np.all(np.round(ds['cross_co_ratio'].data[0, 500]) == 34.0)
    assert np.all(np.round(ds['signal_return_co_pol'].data[0, 11]) == 11)
    assert np.all(np.round(ds['signal_return_co_pol'].data[0, 500]) == -6)
    test_data.close()
    ds.close()


def test_correct_wind():
    nav = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_NAV)
    nav = act.utils.ship_utils.calc_cog_sog(nav)

    aosmet = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_AOSMET)

    ds = xr.merge([nav, aosmet], compat='override')
    ds = act.corrections.ship.correct_wind(ds)

    assert round(ds['wind_speed_corrected'].values[800]) == 5.0
    assert round(ds['wind_direction_corrected'].values[800]) == 92.0


def test_correct_dl():
    # Test the DL correction script on a PPI dataset eventhough it will
    # mostlikely be used on FPT scans. Doing this to save space with only
    # one datafile in the repo.
    files = act.tests.sample_files.EXAMPLE_DLPPI
    ds = act.io.armfiles.read_netcdf(files)

    new_ds = act.corrections.doppler_lidar.correct_dl(ds, fill_value=np.nan)
    data = new_ds['attenuated_backscatter'].values
    np.testing.assert_almost_equal(np.nansum(data), -186479.83, decimal=0.1)

    new_ds = act.corrections.doppler_lidar.correct_dl(ds, range_normalize=False)
    data = new_ds['attenuated_backscatter'].values
    np.testing.assert_almost_equal(np.nansum(data), -200886.0, decimal=0.1)


def test_correct_rl():
    # Using ceil data in RL place to save memory
    files = act.tests.sample_files.EXAMPLE_RL1
    ds = act.io.armfiles.read_netcdf(files)

    ds = act.corrections.raman_lidar.correct_rl(ds, range_normalize_log_values=True)
    np.testing.assert_almost_equal(
        np.max(ds['depolarization_counts_high'].values), 9.91, decimal=2
    )
    np.testing.assert_almost_equal(
        np.min(ds['depolarization_counts_high'].values), -7.00, decimal=2
    )
    np.testing.assert_almost_equal(
        np.mean(ds['depolarization_counts_high'].values), -1.45, decimal=2
    )
