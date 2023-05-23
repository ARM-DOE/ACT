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
        sonde_ds, temp_name='tdry', td_name='dp', p_name='pres'
    )
    np.testing.assert_allclose(
        sonde_ds['parcel_temperature'].values[0:5],
        [269.85, 269.745276, 269.678006, 269.622444, 269.572331],
        rtol=1e-5,
    )
    assert sonde_ds['parcel_temperature'].attrs['units'] == 'kelvin'
    np.testing.assert_almost_equal(sonde_ds['surface_based_cape'], 0.96, decimal=2)
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
        sonde_ds['lifted_condensation_level_temperature'], -8.07, decimal=2
    )
    assert sonde_ds['lifted_condensation_level_temperature'].attrs['units'] == 'degree_Celsius'
    assert (
        sonde_ds['lifted_condensation_level_temperature'].attrs['long_name']
        == 'Lifted condensation level temperature'
    )
    np.testing.assert_almost_equal(sonde_ds['lifted_condensation_level_pressure'], 927.1, decimal=1)
    assert sonde_ds['lifted_condensation_level_pressure'].attrs['units'] == 'hectopascal'
    assert (
        sonde_ds['lifted_condensation_level_pressure'].attrs['long_name']
        == 'Lifted condensation level pressure'
    )
    sonde_ds.close()


def test_generic_sobel_cbh():
    ceil = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_CEIL1)

    ceil = ceil.resample(time='1min').nearest()
    ceil = act.retrievals.cbh.generic_sobel_cbh(
        ceil,
        variable='backscatter',
        height_dim='range',
        var_thresh=1000.0,
        fill_na=0.0,
        edge_thresh=5,
    )
    cbh = ceil['cbh_sobel_backscatter'].values
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
    ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_IRTSST)
    ds = act.retrievals.irt.sst_from_irt(ds)
    np.testing.assert_almost_equal(ds['sea_surface_temperature'].values[0], 278.901, decimal=3)
    np.testing.assert_almost_equal(ds['sea_surface_temperature'].values[-1], 279.291, decimal=3)
    assert np.round(np.nansum(ds['sea_surface_temperature'].values)).astype(int) == 6699
    ds.close()


def test_calculate_sirs_variable():
    sirs_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_SIRS)
    met_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    ds = act.retrievals.radiation.calculate_dsh_from_dsdh_sdn(sirs_ds)
    assert np.isclose(np.nansum(ds['derived_down_short_hemisp'].values), 61157.71, atol=0.1)

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


def test_calculate_pbl_liu_liang():
    files = glob.glob(act.tests.sample_files.EXAMPLE_TWP_SONDE_20060121)
    files2 = glob.glob(act.tests.sample_files.EXAMPLE_SONDE1)
    files += files2
    files.sort()

    pblht = []
    pbl_regime = []
    for i, r in enumerate(files):
        ds = act.io.armfiles.read_netcdf(r)
        ds['tdry'].attrs['units'] = 'degree_Celsius'
        ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds, smooth_height=10)
        pblht.append(float(ds['pblht_liu_liang'].values))
        pbl_regime.append(ds['pblht_regime_liu_liang'].values)

    assert pbl_regime == ['NRL', 'NRL', 'NRL', 'NRL', 'NRL']
    np.testing.assert_array_almost_equal(pblht, [1038.4, 1079.0, 282.0, 314.0, 643.0], decimal=1)

    ds = act.io.armfiles.read_netcdf(files[1])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds, land_parameter=False)
    np.testing.assert_almost_equal(ds['pblht_liu_liang'].values, 784, decimal=1)

    ds = act.io.armfiles.read_netcdf(files[-2:])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    with np.testing.assert_raises(ValueError):
        ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds)

    ds = act.io.armfiles.read_netcdf(files[0])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    temp = ds['tdry'].values
    temp[10:20] = 19.0
    temp[0:10] = -10
    ds['tdry'].values = temp
    ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds, land_parameter=False)
    assert ds['pblht_regime_liu_liang'].values == 'SBL'

    with np.testing.assert_raises(ValueError):
        ds2 = ds.where(ds['alt'].load() < 1000.0, drop=True)
        ds2 = act.retrievals.sonde.calculate_pbl_liu_liang(ds2, smooth_height=15)

    with np.testing.assert_raises(ValueError):
        ds2 = ds.where(ds['pres'].load() < 200.0, drop=True)
        ds2 = act.retrievals.sonde.calculate_pbl_liu_liang(ds2, smooth_height=15)

    with np.testing.assert_raises(ValueError):
        temp[0:5] = -40
        ds['tdry'].values = temp
        ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds)

    ds = act.io.armfiles.read_netcdf(files[0])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    temp = ds['tdry'].values
    temp[20:50] = 100.0
    ds['tdry'].values = temp
    with np.testing.assert_raises(ValueError):
        ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds)


def test_calculate_heffter_pbl():
    files = sorted(glob.glob(act.tests.sample_files.EXAMPLE_TWP_SONDE_20060121))
    ds = act.io.armfiles.read_netcdf(files[2])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    ds = act.retrievals.sonde.calculate_pbl_heffter(ds)
    assert ds['pblht_heffter'].values == 960.0
    np.testing.assert_almost_equal(ds['atm_pres_ss'].values[1], 994.9, 1)
    np.testing.assert_almost_equal(ds['potential_temperature_ss'].values[4], 298.4, 1)
    assert np.sum(ds['bottom_inversion'].values) == 7426
    assert np.sum(ds['top_inversion'].values) == 7903


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason='PySP2 is not installed.')
def test_sp2_waveform_stats():
    my_sp2b = act.io.read_sp2(act.tests.EXAMPLE_SP2B)
    my_ini = act.tests.EXAMPLE_INI
    my_binary = act.qc.get_waveform_statistics(my_sp2b, my_ini, parallel=False)
    assert my_binary.PkHt_ch1.max() == 62669.4
    np.testing.assert_almost_equal(np.nanmax(my_binary.PkHt_ch0.values), 98708.92915295, decimal=1)
    np.testing.assert_almost_equal(np.nanmax(my_binary.PkHt_ch4.values), 65088.39598033, decimal=1)


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason='PySP2 is not installed.')
def test_sp2_psds():
    my_sp2b = act.io.read_sp2(act.tests.EXAMPLE_SP2B)
    my_ini = act.tests.EXAMPLE_INI
    my_binary = act.qc.get_waveform_statistics(my_sp2b, my_ini, parallel=False)
    my_hk = act.io.read_hk_file(act.tests.EXAMPLE_HK)
    my_binary = act.retrievals.calc_sp2_diams_masses(my_binary)
    ScatRejectKey = my_binary['ScatRejectKey'].values
    assert np.nanmax(my_binary['ScatDiaBC50'].values[ScatRejectKey == 0]) < 1000.0
    my_psds = act.retrievals.process_sp2_psds(my_binary, my_hk, my_ini)
    np.testing.assert_almost_equal(my_psds['NumConcIncan'].max(), 0.95805343)
