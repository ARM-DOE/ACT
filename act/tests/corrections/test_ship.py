import xarray as xr

import act


def test_correct_wind():
    nav = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_NAV)
    nav = act.utils.ship_utils.calc_cog_sog(nav)

    aosmet = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_AOSMET)

    ds = xr.merge([nav, aosmet], compat='override')
    ds = act.corrections.ship.correct_wind(ds)

    assert round(ds['wind_speed_corrected'].values[800]) == 5.0
    assert round(ds['wind_direction_corrected'].values[800]) == 92.0
