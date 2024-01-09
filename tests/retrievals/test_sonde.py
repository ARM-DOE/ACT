import numpy as np

import act


def test_get_stability_indices():
    sonde_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_SONDE1)
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


def test_calculate_precipitable_water():
    sonde_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_SONDE1)
    assert sonde_ds['tdry'].units == 'C', 'Temperature must be in Celsius'
    assert sonde_ds['rh'].units == '%', 'Relative Humidity must be a percentage'
    assert sonde_ds['pres'].units == 'hPa', 'Pressure must be in hPa'
    pwv_data = act.retrievals.calculate_precipitable_water(
        sonde_ds, temp_name='tdry', rh_name='rh', pres_name='pres'
    )
    np.testing.assert_almost_equal(pwv_data, 0.8028, decimal=3)
    sonde_ds.close()


def test_calculate_pbl_liu_liang():
    files = act.tests.sample_files.EXAMPLE_TWP_SONDE_20060121.copy()
    files2 = act.tests.sample_files.EXAMPLE_SONDE1
    files.append(files2)
    files.sort()

    pblht = []
    pbl_regime = []
    for file in files:
        ds = act.io.arm.read_arm_netcdf(file)
        ds['tdry'].attrs['units'] = 'degree_Celsius'
        ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds, smooth_height=10)
        pblht.append(float(ds['pblht_liu_liang'].values))
        pbl_regime.append(ds['pblht_regime_liu_liang'].values)

    assert pbl_regime == ['NRL', 'NRL', 'NRL', 'NRL', 'NRL']
    np.testing.assert_array_almost_equal(pblht, [1038.4, 1079.0, 282.0, 314.0, 643.0], decimal=1)

    ds = act.io.arm.read_arm_netcdf(files[1])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds, land_parameter=False)
    np.testing.assert_almost_equal(ds['pblht_liu_liang'].values, 784, decimal=1)

    ds = act.io.arm.read_arm_netcdf(files[-2:])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    with np.testing.assert_raises(ValueError):
        ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds)

    ds = act.io.arm.read_arm_netcdf(files[0])
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

    ds = act.io.arm.read_arm_netcdf(files[0])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    temp = ds['tdry'].values
    temp[20:50] = 100.0
    ds['tdry'].values = temp
    with np.testing.assert_raises(ValueError):
        ds = act.retrievals.sonde.calculate_pbl_liu_liang(ds)


def test_calculate_heffter_pbl():
    files = act.tests.sample_files.EXAMPLE_TWP_SONDE_20060121.copy()
    files.sort()
    ds = act.io.arm.read_arm_netcdf(files[2])
    ds['tdry'].attrs['units'] = 'degree_Celsius'
    ds = act.retrievals.sonde.calculate_pbl_heffter(ds)
    assert ds['pblht_heffter'].values == 960.0
    np.testing.assert_almost_equal(ds['atm_pres_ss'].values[1], 994.9, 1)
    np.testing.assert_almost_equal(ds['potential_temperature_ss'].values[4], 298.4, 1)
    assert np.sum(ds['bottom_inversion'].values) == 7426
    assert np.sum(ds['top_inversion'].values) == 7903
