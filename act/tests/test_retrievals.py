' Unit tests for the ACT retrievals module. ' ''

import glob

import numpy as np
import pytest
import xarray as xr

import act

try:
    import pysp2
    PYSP2_AVAILABLE = True
except ImportError:
    PYSP2_AVAILABLE = False


def test_get_stability_indices():
    sonde_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_SONDE1)
    sonde_ds = act.retrievals.calculate_stability_indicies(
        sonde_ds, temp_name='tdry', td_name='dp', p_name='pres', rh_name='rh')
    np.testing.assert_allclose(
        sonde_ds['parcel_temperature'].values[0:5],
        [269.85, 269.745276, 269.678006, 269.622444, 269.572331],
        rtol=1e-5)
    assert sonde_ds['parcel_temperature'].attrs['units'] == 'kelvin'
    np.testing.assert_almost_equal(sonde_ds['surface_based_cape'], 0.62, decimal=2)
    assert sonde_ds['surface_based_cape'].attrs['units'] == 'J/kg'
    assert sonde_ds['surface_based_cape'].attrs['long_name'] == 'Surface-based CAPE'
    np.testing.assert_almost_equal(sonde_ds['surface_based_cin'], 0.000, decimal=3)
    assert sonde_ds['surface_based_cin'].attrs['units'] == 'J/kg'
    assert sonde_ds['surface_based_cin'].attrs['long_name'] == 'Surface-based CIN'
    np.testing.assert_almost_equal(sonde_ds['most_unstable_cape'], 0.000, decimal=3)
    assert sonde_ds['most_unstable_cape'].attrs['units'] == 'J/kg'
    assert sonde_ds['most_unstable_cape'].attrs['long_name'] == 'Most unstable CAPE'
    np.testing.assert_almost_equal(sonde_ds['most_unstable_cin'], 0.000, decimal=3)
    assert sonde_ds['most_unstable_cin'].attrs['units'] == 'J/kg'
    assert sonde_ds['most_unstable_cin'].attrs['long_name'] == 'Most unstable CIN'

    np.testing.assert_almost_equal(sonde_ds['lifted_index'], 28.4, decimal=1)
    assert sonde_ds['lifted_index'].attrs['units'] == 'kelvin'
    assert sonde_ds['lifted_index'].attrs['long_name'] == 'Lifted index'
    np.testing.assert_equal(sonde_ds['level_of_free_convection'], np.array(np.nan))
    assert sonde_ds['level_of_free_convection'].attrs['units'] == 'hectopascal'
    assert sonde_ds['level_of_free_convection'].attrs['long_name'] == 'Level of free convection'
    np.testing.assert_almost_equal(
        sonde_ds['lifted_condensation_level_temperature'], -8.07, decimal=2)
    assert sonde_ds['lifted_condensation_level_temperature'].attrs['units'] == 'degree_Celsius'
    assert (
        sonde_ds['lifted_condensation_level_temperature'].attrs['long_name']
        == 'Lifted condensation level temperature')
    np.testing.assert_almost_equal(
        sonde_ds['lifted_condensation_level_pressure'], 927.1, decimal=1)
    assert sonde_ds['lifted_condensation_level_pressure'].attrs['units'] == 'hectopascal'
    assert (
        sonde_ds['lifted_condensation_level_pressure'].attrs['long_name']
        == 'Lifted condensation level pressure')
    sonde_ds.close()


def test_generic_sobel_cbh():
    ceil = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_CEIL1)

    ceil = ceil.resample(time='1min').nearest()
    ceil = act.retrievals.cbh.generic_sobel_cbh(
        ceil, variable='backscatter', height_dim='range', var_thresh=1000.0, fill_na=0
    )
    cbh = ceil['cbh_sobel'].values
    assert cbh[500] == 615.0
    assert cbh[1000] == 555.0
    ceil.close()


def test_calculate_precipitable_water():
    sonde_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_SONDE1)
    assert sonde_ds['tdry'].units == 'C', 'Temperature must be in Celsius'
    assert sonde_ds['rh'].units == '%', 'Relative Humidity must be a percentage'
    assert sonde_ds['pres'].units == 'hPa', 'Pressure must be in hPa'
    pwv_data = act.retrievals.calculate_precipitable_water(
        sonde_ds, temp_name='tdry', rh_name='rh', pres_name='pres'
    )
    np.testing.assert_almost_equal(pwv_data, 0.8028, decimal=3)
    sonde_ds.close()


def test_doppler_lidar_winds():
    # Process a single file
    dl_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_DLPPI)
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds, intensity_name='intensity')
    assert np.round(np.nansum(result['wind_speed'].values)).astype(int) == 1570
    assert np.round(np.nansum(result['wind_direction'].values)).astype(int) == 32635
    assert result['wind_speed'].attrs['units'] == 'm/s'
    assert result['wind_direction'].attrs['units'] == 'degree'
    assert result['height'].attrs['units'] == 'm'
    dl_ds.close()

    # Process multiple files
    dl_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_DLPPI_MULTI)
    del dl_ds['range'].attrs['units']
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds)
    assert np.round(np.nansum(result['wind_speed'].values)).astype(int) == 2854
    assert np.round(np.nansum(result['wind_direction'].values)).astype(int) == 64986
    dl_ds.close()


def test_aeri2irt():
    aeri_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_AERI)
    aeri_ds = act.retrievals.aeri.aeri2irt(aeri_ds)
    assert np.round(np.nansum(aeri_ds['aeri_irt_equiv_temperature'].values)).astype(int) == 17372
    np.testing.assert_almost_equal(
        aeri_ds['aeri_irt_equiv_temperature'].values[7], 286.081, decimal=3
    )
    np.testing.assert_almost_equal(
        aeri_ds['aeri_irt_equiv_temperature'].values[-10], 285.366, decimal=3
    )
    aeri_ds.close()
    del aeri_ds


def test_sst():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_IRTSST)
    obj = act.retrievals.irt.sst_from_irt(obj)
    np.testing.assert_almost_equal(obj['sea_surface_temperature'].values[0], 278.901, decimal=3)
    np.testing.assert_almost_equal(obj['sea_surface_temperature'].values[-1], 279.291, decimal=3)
    assert np.round(np.nansum(obj['sea_surface_temperature'].values)).astype(int) == 6699
    obj.close()


def test_calculate_sirs_variable():
    sirs_object = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_SIRS)
    met_object = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    obj = act.retrievals.radiation.calculate_dsh_from_dsdh_sdn(sirs_object)
    assert np.isclose(np.nansum(obj['derived_down_short_hemisp'].values), 61157.71, atol=0.1)

    obj = act.retrievals.radiation.calculate_irradiance_stats(
        obj,
        variable='derived_down_short_hemisp',
        variable2='down_short_hemisp',
        threshold=60,
    )

    assert np.isclose(np.nansum(obj['diff_derived_down_short_hemisp'].values), 1335.12, atol=0.1)
    assert np.isclose(np.nansum(obj['ratio_derived_down_short_hemisp'].values), 400.31, atol=0.1)

    obj = act.retrievals.radiation.calculate_net_radiation(obj, smooth=30)
    assert np.ceil(np.nansum(obj['net_radiation'].values)) == 21915
    assert np.ceil(np.nansum(obj['net_radiation_smoothed'].values)) == 22316

    obj = act.retrievals.radiation.calculate_longwave_radiation(
        obj,
        temperature_var='temp_mean',
        vapor_pressure_var='vapor_pressure_mean',
        met_obj=met_object,
    )
    assert np.ceil(obj['monteith_clear'].values[25]) == 239
    assert np.ceil(obj['monteith_cloudy'].values[30]) == 318
    assert np.ceil(obj['prata_clear'].values[35]) == 234

    new_obj = xr.merge([sirs_object, met_object], compat='override')
    obj = act.retrievals.radiation.calculate_longwave_radiation(
        new_obj, temperature_var='temp_mean', vapor_pressure_var='vapor_pressure_mean'
    )
    assert np.ceil(obj['monteith_clear'].values[25]) == 239
    assert np.ceil(obj['monteith_cloudy'].values[30]) == 318
    assert np.ceil(obj['prata_clear'].values[35]) == 234

    sirs_object.close()
    met_object.close()


def test_calculate_pbl_liu_liang():
    files = glob.glob(act.tests.sample_files.EXAMPLE_TWP_SONDE_20060121)
    files2 = glob.glob(act.tests.sample_files.EXAMPLE_SONDE1)
    files += files2
    files.sort()

    pblht = []
    pbl_regime = []
    for i, r in enumerate(files):
        obj = act.io.armfiles.read_netcdf(r)
        obj['tdry'].attrs['units'] = 'degree_Celsius'
        obj = act.retrievals.sonde.calculate_pbl_liu_liang(obj, smooth_height=10)
        pblht.append(float(obj['pblht_liu_liang'].values))
        pbl_regime.append(obj['pblht_regime_liu_liang'].values)

    assert pbl_regime == ['NRL', 'NRL', 'NRL', 'NRL', 'NRL']
    np.testing.assert_array_almost_equal(pblht, [847.5, 858.2, 184.8, 197.7, 443.2], decimal=1)

    obj = act.io.armfiles.read_netcdf(files[1])
    obj['tdry'].attrs['units'] = 'degree_Celsius'
    obj = act.retrievals.sonde.calculate_pbl_liu_liang(obj, land_parameter=False)
    np.testing.assert_almost_equal(obj['pblht_liu_liang'].values, 733.6, decimal=1)

    obj = act.io.armfiles.read_netcdf(files[-2:])
    obj['tdry'].attrs['units'] = 'degree_Celsius'
    with np.testing.assert_raises(ValueError):
        obj = act.retrievals.sonde.calculate_pbl_liu_liang(obj)

    obj = act.io.armfiles.read_netcdf(files[0])
    obj['tdry'].attrs['units'] = 'degree_Celsius'
    temp = obj['tdry'].values
    temp[10:20] = 19.0
    temp[0:10] = -10
    obj['tdry'].values = temp
    obj = act.retrievals.sonde.calculate_pbl_liu_liang(obj, land_parameter=False)
    assert obj['pblht_regime_liu_liang'].values == 'SBL'

    with np.testing.assert_raises(ValueError):
        obj2 = obj.where(obj['alt'] < 1000.0, drop=True)
        obj2 = act.retrievals.sonde.calculate_pbl_liu_liang(obj2, smooth_height=15)

    with np.testing.assert_raises(ValueError):
        obj2 = obj.where(obj['pres'] < 200.0, drop=True)
        obj2 = act.retrievals.sonde.calculate_pbl_liu_liang(obj2, smooth_height=15)

    with np.testing.assert_raises(ValueError):
        temp[0:5] = -40
        obj['tdry'].values = temp
        obj = act.retrievals.sonde.calculate_pbl_liu_liang(obj)

    obj = act.io.armfiles.read_netcdf(files[0])
    obj['tdry'].attrs['units'] = 'degree_Celsius'
    temp = obj['tdry'].values
    temp[20:50] = 100.0
    obj['tdry'].values = temp
    with np.testing.assert_raises(ValueError):
        obj = act.retrievals.sonde.calculate_pbl_liu_liang(obj)


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason="PySP2 is not installed.")
def test_sp2_waveform_stats():
    my_sp2b = act.io.read_sp2(act.tests.EXAMPLE_SP2B)
    my_ini = act.tests.EXAMPLE_INI
    my_binary = act.qc.get_waveform_statistics(my_sp2b, my_ini, parallel=False)
    assert my_binary.PkHt_ch1.max() == 62669.4
    np.testing.assert_almost_equal(np.nanmax(my_binary.PkHt_ch0.values), 98708.92915295, decimal=1)
    np.testing.assert_almost_equal(np.nanmax(my_binary.PkHt_ch4.values), 54734.05714286, decimal=1)


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason="PySP2 is not installed.")
def test_sp2_psds():
    my_sp2b = act.io.read_sp2(act.tests.EXAMPLE_SP2B)
    my_ini = act.tests.EXAMPLE_INI
    my_binary = act.qc.get_waveform_statistics(my_sp2b, my_ini, parallel=False)
    my_hk = act.io.read_hk_file(act.tests.EXAMPLE_HK)
    my_binary = act.retrievals.calc_sp2_diams_masses(my_binary)
    ScatRejectKey = my_binary['ScatRejectKey'].values
    assert np.nanmax(
        my_binary['ScatDiaBC50'].values[ScatRejectKey == 0]) < 1000.0
    my_psds = act.retrievals.process_sp2_psds(my_binary, my_hk, my_ini)
    np.testing.assert_almost_equal(my_psds['NumConcIncan'].max(), 0.95805343)
