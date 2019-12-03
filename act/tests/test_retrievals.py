import act
import numpy as np


def test_get_stability_indices():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_SONDE1)

    sonde_ds = act.retrievals.calculate_stability_indicies(
        sonde_ds, temp_name="tdry", td_name="dp", p_name="pres")
    assert sonde_ds["lifted_index"].units == "kelvin"
    np.testing.assert_almost_equal(sonde_ds["lifted_index"], 28.4639, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["most_unstable_cin"], 0.000, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["surface_based_cin"], 0.000, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["most_unstable_cape"], 0.000, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["surface_based_cape"], 1.628, decimal=3)
    np.testing.assert_almost_equal(
        sonde_ds["lifted_condensation_level_pressure"], 927.143, decimal=3)
    np.testing.assert_almost_equal(
        sonde_ds["lifted_condensation_level_temperature"], -8.079, decimal=3)
    sonde_ds.close()


def test_generic_sobel_cbh():
    ceil = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_CEIL1)

    ceil = ceil.resample(time='1min').nearest()
    ceil = act.retrievals.cbh.generic_sobel_cbh(ceil, variable='backscatter',
                                                height_dim='range', var_thresh=1000.,
                                                fill_na=0)
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
    pwv_data = act.retrievals.pwv_calc.calculate_precipitable_water(
        sonde_ds, temp_name='tdry', rh_name='rh', pres_name='pres')
    np.testing.assert_almost_equal(pwv_data, 0.8028, decimal=3)
    sonde_ds.close()


def test_doppler_lidar_winds():
    dl_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_DLPPI)
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds, intensity_name='intensity')
    assert np.round(np.nansum(result['wind_speed'].values)).astype(int) == 1570
    assert np.round(np.nansum(result['wind_direction'].values)).astype(int) == 32635
    assert result['wind_speed'].attrs['units'] == 'm/s'
    assert result['wind_direction'].attrs['units'] == 'degree'
    assert result['height'].attrs['units'] == 'm'
    dl_ds.close()
    del dl_ds
    del result

def test_calculate_dqr_times():
    ebbr1_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_EBBR1)
    ebbr2_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_EBBR2)
    aos_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_AOSPURGE)
    ebbr1_result = act.retrievals.dqr_times.calculate_dqr_times(
        ebbr1_ds, variable=['soil_temp_1'], threshold=2)
    ebbr2_result = act.retrievals.dqr_times.calculate_dqr_times(
        ebbr2_ds, variable=['rh_bottom_fraction', 'rh_top_fraction'],
        qc_bit=3, threshold=2)
    aos_result = act.retrievals.dqr_times.calculate_dqr_times(
        aos_ds, aos_purge=True, threshold=30)
    assert ebbr1_result == [('2019-11-25 02:00:00', '2019-11-25 04:30:00')]
    assert ebbr2_result == [('2019-11-30 00:00:00', '2019-11-30 11:00:00')]
    assert aos_result == [('2019-10-16 00:08:04', '2019-10-16 00:10:16'),
                         ('2019-10-16 00:29:27', '2019-10-16 00:44:35'),
                         ('2019-10-16 00:51:28', '2019-10-16 00:53:51'),
                         ('2019-10-16 00:59:39', '2019-10-16 01:00:10')]
    ebbr1_ds.close()
    ebbr2_ds.close()
    aos_ds.close()