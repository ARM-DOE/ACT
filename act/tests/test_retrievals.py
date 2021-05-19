" Unit tests for the ACT retrievals module. """

import act
import numpy as np
import xarray as xr


def test_get_stability_indices():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_SONDE1)

    try:
        sonde_ds = act.retrievals.calculate_stability_indicies(
            sonde_ds, temp_name="tdry", td_name="dp", p_name="pres")
        metpy = True
    except ImportError:
        metpy = False

    if metpy is True:
        np.testing.assert_allclose(
            sonde_ds["parcel_temperature"].values[0:5],
            [269.85000005, 269.74530704, 269.67805708,
             269.62251119, 269.57241322], rtol=1e-5)
        assert sonde_ds["parcel_temperature"].attrs["units"] == "kelvin"
        np.testing.assert_almost_equal(
            sonde_ds["surface_based_cape"], 1.62, decimal=2)
        assert sonde_ds["surface_based_cape"].attrs["units"] == "J/kg"
        assert sonde_ds[
            "surface_based_cape"].attrs["long_name"] == "Surface-based CAPE"
        np.testing.assert_almost_equal(
            sonde_ds["surface_based_cin"], 0.000, decimal=3)
        assert sonde_ds["surface_based_cin"].attrs["units"] == "J/kg"
        assert sonde_ds[
            "surface_based_cin"].attrs["long_name"] == "Surface-based CIN"
        np.testing.assert_almost_equal(
            sonde_ds["most_unstable_cape"], 0.000, decimal=3)
        assert sonde_ds["most_unstable_cape"].attrs["units"] == "J/kg"
        assert sonde_ds[
            "most_unstable_cape"].attrs["long_name"] == "Most unstable CAPE"
        np.testing.assert_almost_equal(
            sonde_ds["most_unstable_cin"], 0.000, decimal=3)
        assert sonde_ds["most_unstable_cin"].attrs["units"] == "J/kg"
        assert sonde_ds[
            "most_unstable_cin"].attrs["long_name"] == "Most unstable CIN"

        np.testing.assert_almost_equal(
            sonde_ds["lifted_index"], 28.4, decimal=1)
        assert sonde_ds["lifted_index"].attrs["units"] == "kelvin"
        assert sonde_ds["lifted_index"].attrs["long_name"] == "Lifted index"
        np.testing.assert_equal(
            sonde_ds["level_of_free_convection"], np.array(np.nan))
        assert sonde_ds[
            "level_of_free_convection"].attrs["units"] == "hectopascal"
        assert sonde_ds[
            "level_of_free_convection"].attrs[
                "long_name"] == "Level of free convection"
        np.testing.assert_almost_equal(
            sonde_ds["lifted_condensation_level_temperature"],
            -8.07, decimal=2)
        assert sonde_ds[
            "lifted_condensation_level_temperature"].attrs[
                "units"] == "degree_Celsius"
        assert sonde_ds[
            "lifted_condensation_level_temperature"].attrs[
                "long_name"] == "Lifted condensation level temperature"
        np.testing.assert_almost_equal(
            sonde_ds["lifted_condensation_level_pressure"], 927.1, decimal=1)
        assert sonde_ds[
            "lifted_condensation_level_pressure"].attrs[
                "units"] == "hectopascal"
        assert sonde_ds[
            "lifted_condensation_level_pressure"].attrs[
                "long_name"] == "Lifted condensation level pressure"
    sonde_ds.close()


def test_generic_sobel_cbh():
    ceil = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_CEIL1)

    ceil = ceil.resample(time='1min').nearest()
    ceil = act.retrievals.cbh.generic_sobel_cbh(
        ceil, variable='backscatter', height_dim='range',
        var_thresh=1000., fill_na=0)
    cbh = ceil['cbh_sobel'].values
    assert cbh[500] == 615.
    assert cbh[1000] == 555.
    ceil.close()


def test_calculate_precipitable_water():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_SONDE1)
    assert sonde_ds["tdry"].units == "C", "Temperature must be in Celsius"
    assert sonde_ds["rh"].units == "%", "Relative Humidity must be a percentage"
    assert sonde_ds["pres"].units == "hPa", "Pressure must be in hPa"
    pwv_data = act.retrievals.calculate_precipitable_water(
        sonde_ds, temp_name='tdry', rh_name='rh', pres_name='pres')
    np.testing.assert_almost_equal(pwv_data, 0.8028, decimal=3)
    sonde_ds.close()


def test_doppler_lidar_winds():
    # Process a single file
    dl_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_DLPPI)
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(
        dl_ds, intensity_name='intensity')
    assert np.round(
        np.nansum(result['wind_speed'].values)).astype(int) == 1570
    assert np.round(
        np.nansum(result['wind_direction'].values)).astype(int) == 32635
    assert result['wind_speed'].attrs['units'] == 'm/s'
    assert result['wind_direction'].attrs['units'] == 'degree'
    assert result['height'].attrs['units'] == 'm'
    dl_ds.close()

    # Process multiple files
    dl_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_DLPPI_MULTI)
    del dl_ds['range'].attrs['units']
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds)
    assert np.round(
        np.nansum(result['wind_speed'].values)).astype(int) == 64419
    assert np.round(
        np.nansum(result['wind_direction'].values)).astype(int) == 733627
    dl_ds.close()


def test_aeri2irt():
    aeri_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_AERI)
    aeri_ds = act.retrievals.aeri.aeri2irt(aeri_ds)
    assert np.round(
        np.nansum(
            aeri_ds['aeri_irt_equiv_temperature'].values)).astype(int) == 17372
    np.testing.assert_almost_equal(
        aeri_ds['aeri_irt_equiv_temperature'].values[7], 286.081, decimal=3)
    np.testing.assert_almost_equal(
        aeri_ds['aeri_irt_equiv_temperature'].values[-10], 285.366, decimal=3)
    aeri_ds.close()
    del aeri_ds


def test_sst():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_IRTSST)
    obj = act.retrievals.irt.sst_from_irt(obj)
    np.testing.assert_almost_equal(
        obj['sea_surface_temperature'].values[0], 278.901, decimal=3)
    np.testing.assert_almost_equal(
        obj['sea_surface_temperature'].values[-1], 279.291, decimal=3)
    assert np.round(
        np.nansum(obj['sea_surface_temperature'].values)).astype(int) == 6699
    obj.close()


def test_calculate_sirs_variable():
    sirs_object = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_SIRS)
    met_object = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_MET1)

    obj = act.retrievals.radiation.calculate_dsh_from_dsdh_sdn(sirs_object)
    assert np.isclose(np.nansum(obj['derived_down_short_hemisp'].values), 60350, atol=1)

    obj = act.retrievals.radiation.calculate_irradiance_stats(
        obj, variable='derived_down_short_hemisp',
        variable2='down_short_hemisp', threshold=60)
    assert np.isclose(np.nansum(obj['diff_derived_down_short_hemisp'].values), 527, atol=1)
    assert np.isclose(np.nansum(obj['ratio_derived_down_short_hemisp'].values), 392, atol=1)

    obj = act.retrievals.radiation.calculate_net_radiation(obj, smooth=30)
    assert np.ceil(np.nansum(obj['net_radiation'].values)) == 21915
    assert np.ceil(np.nansum(obj['net_radiation_smoothed'].values)) == 22316

    obj = act.retrievals.radiation.calculate_longwave_radiation(
        obj, temperature_var='temp_mean',
        vapor_pressure_var='vapor_pressure_mean',
        met_obj=met_object)
    assert np.ceil(obj['monteith_clear'].values[25]) == 239
    assert np.ceil(obj['monteith_cloudy'].values[30]) == 318
    assert np.ceil(obj['prata_clear'].values[35]) == 234

    new_obj = xr.merge([sirs_object, met_object], compat='override')
    obj = act.retrievals.radiation.calculate_longwave_radiation(
        new_obj, temperature_var='temp_mean',
        vapor_pressure_var='vapor_pressure_mean')
    assert np.ceil(obj['monteith_clear'].values[25]) == 239
    assert np.ceil(obj['monteith_cloudy'].values[30]) == 318
    assert np.ceil(obj['prata_clear'].values[35]) == 234

    sirs_object.close()
    met_object.close()
