import numpy as np
import xarray as xr

import act


def test_calculate_sirs_variable():
    sirs_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_SIRS)
    met_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MET1)
    mfrsr_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MFRSR)

    elevation, _, _ = act.utils.geo_utils.get_solar_azimuth_elevation(
        sirs_ds['lat'].values, sirs_ds['lon'].values, sirs_ds['time'].values
    )
    sirs_ds['solar_zenith_angle'] = xr.DataArray(
        90.0 - elevation,
        dims=['time'],
        attrs={
            'long_name': 'Solar zenith angle',
            'units': 'degrees',
        },
    )

    # old
    ds = act.retrievals.radiation.calculate_dsh_from_dsdh_sdn(sirs_ds)
    assert np.isclose(np.nansum(ds['derived_down_short_hemisp'].values), 61157.71, atol=0.1)

    # new
    ds = act.retrievals.radiation.calculate_ghi_from_dni_dhi(sirs_ds)
    assert np.isclose(np.nansum(ds['derived_down_short_hemisp'].values), 61157.71, atol=0.1)

    # Test with providing solar_zenith variable in Dataset
    ds = act.retrievals.radiation.calculate_ghi_from_dni_dhi(
        sirs_ds, solar_zenith='solar_zenith_angle'
    )
    assert np.isclose(np.nansum(ds['derived_down_short_hemisp'].values), 61157.71, atol=0.1)

    ds = act.retrievals.radiation.calculate_dhi_from_dni_ghi(sirs_ds)
    assert np.isclose(np.nansum(ds['derived_down_short_diffuse_hemisp'].values), 59825.39, atol=0.1)

    # Test with providing solar_zenith variable in Dataset
    ds = act.retrievals.radiation.calculate_dhi_from_dni_ghi(
        sirs_ds, solar_zenith='solar_zenith_angle'
    )
    assert np.isclose(np.nansum(ds['derived_down_short_diffuse_hemisp'].values), 59825.39, atol=0.1)

    # Test direct normal
    mfrsr_ds = act.retrievals.radiation.calculate_dni_from_dhi_ghi(
        mfrsr_ds,
        ghi='hemisp_narrowband_filter3',
        dhi='diffuse_hemisp_narrowband_filter3',
        derived='derived_direct_normal_narrowband_filter3',
    )
    assert np.isclose(
        np.nansum(mfrsr_ds['derived_direct_normal_narrowband_filter3'].values), 2553.92, atol=0.1
    )

    # Test with providing solar_zenith variable in Dataset
    mfrsr_ds = act.retrievals.radiation.calculate_dni_from_dhi_ghi(
        mfrsr_ds,
        ghi='hemisp_narrowband_filter3',
        dhi='diffuse_hemisp_narrowband_filter3',
        derived='derived_direct_normal_narrowband_filter3',
        solar_zenith='solar_zenith_angle',
    )
    assert np.isclose(
        np.nansum(mfrsr_ds['derived_direct_normal_narrowband_filter3'].values), 2553.59, atol=0.1
    )

    ds = act.retrievals.radiation.calculate_irradiance_stats(
        ds,
        variable='derived_down_short_hemisp',
        variable2='down_short_hemisp',
        threshold=60,
    )

    assert np.isclose(np.nansum(ds['diff_derived_down_short_hemisp'].values), 1335.12, atol=0.1)
    assert np.isclose(np.nansum(ds['ratio_derived_down_short_hemisp'].values), 400.31, atol=0.1)

    ds = act.retrievals.radiation.calculate_net_radiation(ds, smooth=30)
    assert np.ceil(np.nansum(ds['net_radiation'].values)) == 21915
    assert np.ceil(np.nansum(ds['net_radiation_smoothed'].values)) == 22316

    ds = act.retrievals.radiation.calculate_longwave_radiation(
        ds,
        temperature_var='temp_mean',
        vapor_pressure_var='vapor_pressure_mean',
        met_ds=met_ds,
    )
    assert np.ceil(ds['monteith_clear'].values[25]) == 239
    assert np.ceil(ds['monteith_cloudy'].values[30]) == 318
    assert np.ceil(ds['prata_clear'].values[35]) == 234

    new_ds = xr.merge([sirs_ds, met_ds], compat='override')
    ds = act.retrievals.radiation.calculate_longwave_radiation(
        new_ds, temperature_var='temp_mean', vapor_pressure_var='vapor_pressure_mean'
    )
    assert np.ceil(ds['monteith_clear'].values[25]) == 239
    assert np.ceil(ds['monteith_cloudy'].values[30]) == 318
    assert np.ceil(ds['prata_clear'].values[35]) == 234

    sirs_ds.close()
    met_ds.close()
