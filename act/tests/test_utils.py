""" Unit tests for ACT utils module. """

from datetime import datetime
import pytz

import act
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import tempfile
from pathlib import Path


def test_dates_between():
    start_date = '20190101'
    end_date = '20190110'
    date_list = act.utils.dates_between(start_date, end_date)
    answer = [datetime(2019, 1, 1),
              datetime(2019, 1, 2),
              datetime(2019, 1, 3),
              datetime(2019, 1, 4),
              datetime(2019, 1, 5),
              datetime(2019, 1, 6),
              datetime(2019, 1, 7),
              datetime(2019, 1, 8),
              datetime(2019, 1, 9),
              datetime(2019, 1, 10)]
    assert date_list == answer


def test_add_in_nan():
    # Make a 1D array with a 4 day gap in the data
    time_list = [np.datetime64(datetime(2019, 1, 1, 1, 0)),
                 np.datetime64(datetime(2019, 1, 1, 1, 1)),
                 np.datetime64(datetime(2019, 1, 1, 1, 2)),
                 np.datetime64(datetime(2019, 1, 1, 1, 8)),
                 np.datetime64(datetime(2019, 1, 1, 1, 9))]
    data = np.linspace(0., 8., 5)

    time_list = xr.DataArray(time_list)
    data = xr.DataArray(data)
    time_filled, data_filled = act.utils.add_in_nan(
        time_list, data)

    assert(data_filled.data[8] == 6.)

    time_answer = [np.datetime64(datetime(2019, 1, 1, 1, 0)),
                   np.datetime64(datetime(2019, 1, 1, 1, 1)),
                   np.datetime64(datetime(2019, 1, 1, 1, 2)),
                   np.datetime64(datetime(2019, 1, 1, 1, 3)),
                   np.datetime64(datetime(2019, 1, 1, 1, 4)),
                   np.datetime64(datetime(2019, 1, 1, 1, 5)),
                   np.datetime64(datetime(2019, 1, 1, 1, 6)),
                   np.datetime64(datetime(2019, 1, 1, 1, 7)),
                   np.datetime64(datetime(2019, 1, 1, 1, 8)),
                   np.datetime64(datetime(2019, 1, 1, 1, 9))]

    assert(time_filled[8].values == time_answer[8])
    assert(time_filled[5].values == time_answer[5])

    time_filled, data_filled = act.utils.add_in_nan(
        time_list[0], data[0])
    assert time_filled.values == time_list[0]
    assert data_filled.values == data[0]


def test_get_missing_value():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    missing = act.utils.data_utils.get_missing_value(
        obj, 'latent_heat_flux', use_FillValue=True, add_if_missing_in_obj=True)
    assert missing == -9999

    obj['latent_heat_flux'].attrs['missing_value'] = -9998
    missing = act.utils.data_utils.get_missing_value(obj, 'latent_heat_flux')
    assert missing == -9998


def test_convert_units():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    data = obj['soil_temp_1'].values
    in_units = obj['soil_temp_1'].attrs['units']
    r_data = act.utils.data_utils.convert_units(data, in_units, 'K')
    assert np.ceil(r_data[0]) == 285

    data = act.utils.data_utils.convert_units(r_data, 'K', 'C')
    assert np.ceil(data[0]) == 12

    try:
        obj.utils.change_units()
    except ValueError as error:
        assert str(error) == "Need to provide 'desired_unit' keyword for .change_units() method"

    desired_unit = 'degF'
    skip_vars = [ii for ii in obj.data_vars if ii.startswith('qc_')]
    obj.utils.change_units(variables=None, desired_unit=desired_unit,
                           skip_variables=skip_vars, skip_standard=True)
    units = []
    for var_name in obj.data_vars:
        try:
            units.append(obj[var_name].attrs['units'])
        except KeyError:
            pass
    indices = [i for i, x in enumerate(units) if x == desired_unit]
    assert indices == [0, 2, 4, 6, 8, 32, 34, 36, 38, 40]

    var_name = 'home_signal_15'
    desired_unit = 'V'
    obj.utils.change_units(var_name, desired_unit, skip_variables='lat')
    assert obj[var_name].attrs['units'] == desired_unit

    var_names = ['home_signal_15', 'home_signal_30']
    obj.utils.change_units(var_names, desired_unit)
    for var_name in var_names:
        assert obj[var_name].attrs['units'] == desired_unit

    obj.close()
    del obj

    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_CEIL1)
    var_name = 'range'
    desired_unit = 'km'
    obj = obj.utils.change_units(var_name, desired_unit)
    assert obj[var_name].attrs['units'] == desired_unit
    assert np.isclose(np.sum(obj[var_name].values), 952.56, atol=0.01)

    obj.close()
    del obj


def test_ts_weighted_average():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET_WILDCARD)
    cf_ds = {'sgpmetE13.b1': {'variable': ['tbrg_precip_total', 'org_precip_rate_mean',
                                           'pwd_precip_rate_mean_1min'],
                              'weight': [0.8, 0.15, 0.05], 'object': obj}}
    data = act.utils.data_utils.ts_weighted_average(cf_ds)

    np.testing.assert_almost_equal(np.sum(data), 84.9, decimal=1)


def test_accum_precip():
    obj = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_MET_WILDCARD)

    obj = act.utils.accumulate_precip(obj, 'tbrg_precip_total')
    dmax = round(np.nanmax(obj['tbrg_precip_total_accumulated']))
    assert dmax == 13.0

    obj = act.utils.accumulate_precip(obj, 'tbrg_precip_total', time_delta=60)
    dmax = round(np.nanmax(obj['tbrg_precip_total_accumulated']))
    assert dmax == 13.0

    obj['tbrg_precip_total'].attrs['units'] = 'mm/hr'
    obj = act.utils.accumulate_precip(obj, 'tbrg_precip_total')
    dmax = np.round(np.nanmax(obj['tbrg_precip_total_accumulated']), 2)
    assert dmax == 0.22


def test_calc_cog_sog():
    obj = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_NAV)

    obj = act.utils.calc_cog_sog(obj)

    cog = obj['course_over_ground'].values
    sog = obj['speed_over_ground'].values

    np.testing.assert_almost_equal(cog[10], 170.987, decimal=3)
    np.testing.assert_almost_equal(sog[15], 0.448, decimal=3)

    obj = obj.rename({'lat': 'latitude', 'lon': 'longitude'})
    obj = act.utils.calc_cog_sog(obj)
    np.testing.assert_almost_equal(cog[10], 170.987, decimal=3)
    np.testing.assert_almost_equal(sog[15], 0.448, decimal=3)


def test_destination_azimuth_distance():
    lat = 37.1509
    lon = -98.362
    lat2, lon2 = act.utils.destination_azimuth_distance(lat, lon, 180., 100)

    np.testing.assert_almost_equal(lat2, 37.150, decimal=3)
    np.testing.assert_almost_equal(lon2, -98.361, decimal=3)


def test_calculate_dqr_times():
    ebbr1_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_EBBR1)
    ebbr2_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_EBBR2)
    brs_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_BRS)
    ebbr1_result = act.utils.calculate_dqr_times(
        ebbr1_ds, variable=['soil_temp_1'], threshold=2)
    ebbr2_result = act.utils.calculate_dqr_times(
        ebbr2_ds, variable=['rh_bottom_fraction'],
        qc_bit=3, threshold=2)
    ebbr3_result = act.utils.calculate_dqr_times(
        ebbr2_ds, variable=['rh_bottom_fraction'],
        qc_bit=3)
    brs_result = act.utils.calculate_dqr_times(
        brs_ds, variable='down_short_hemisp_min', qc_bit=2, threshold=30)
    assert ebbr1_result == [('2019-11-25 02:00:00', '2019-11-25 04:30:00')]
    assert ebbr2_result == [('2019-11-30 00:00:00', '2019-11-30 11:00:00')]
    assert brs_result == [('2019-07-05 01:57:00', '2019-07-05 11:07:00')]
    assert ebbr3_result is None
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_file = Path(tmpdirname)
        brs_result = act.utils.calculate_dqr_times(brs_ds, variable='down_short_hemisp_min',
                                                   qc_bit=2, threshold=30, txt_path=str(write_file))

    brs_result = act.utils.calculate_dqr_times(brs_ds, variable='down_short_hemisp_min',
                                               qc_bit=2, threshold=30, return_missing=False)
    assert len(brs_result[0]) == 2

    ebbr1_ds.close()
    ebbr2_ds.close()
    brs_ds.close()


def test_decode_present_weather():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET1)
    obj = act.utils.decode_present_weather(obj, variable='pwd_pw_code_inst')

    data = obj['pwd_pw_code_inst_decoded'].values
    result = 'No significant weather observed'
    assert data[0] == result
    assert data[100] == result
    assert data[600] == result

    np.testing.assert_raises(ValueError, act.utils.inst_utils.decode_present_weather, obj)
    np.testing.assert_raises(ValueError, act.utils.inst_utils.decode_present_weather,
                             obj, variable='temp_temp')


def test_datetime64_to_datetime():
    time_datetime = [datetime(2019, 1, 1, 1, 0),
                     datetime(2019, 1, 1, 1, 1),
                     datetime(2019, 1, 1, 1, 2),
                     datetime(2019, 1, 1, 1, 3),
                     datetime(2019, 1, 1, 1, 4)]

    time_datetime64 = [np.datetime64(datetime(2019, 1, 1, 1, 0)),
                       np.datetime64(datetime(2019, 1, 1, 1, 1)),
                       np.datetime64(datetime(2019, 1, 1, 1, 2)),
                       np.datetime64(datetime(2019, 1, 1, 1, 3)),
                       np.datetime64(datetime(2019, 1, 1, 1, 4))]

    time_datetime64_to_datetime = act.utils.datetime_utils.datetime64_to_datetime(time_datetime64)
    assert time_datetime == time_datetime64_to_datetime


def test_create_pyart_obj():
    try:
        obj = act.io.mpl.read_sigma_mplv5(act.tests.EXAMPLE_SIGMA_MPLV5)
    except Exception:
        return

    radar = act.utils.create_pyart_obj(obj, range_var='range')
    variables = list(radar.fields)
    assert 'nrb_copol' in variables
    assert 'nrb_crosspol' in variables
    assert radar.sweep_start_ray_index['data'][-1] == 67
    assert radar.sweep_end_ray_index['data'][-1] == 101
    assert radar.fixed_angle['data'] == 2.0
    assert radar.scan_type == 'ppi'
    assert radar.sweep_mode['data'] == 'ppi'
    np.testing.assert_allclose(
        radar.sweep_number['data'][-3:],
        [1., 1., 1.])
    np.testing.assert_allclose(
        radar.sweep_number['data'][0:3],
        [0., 0., 0.])

    # coordinates
    np.testing.assert_allclose(
        radar.azimuth['data'][0:5],
        [-95., -92.5, -90., -87.5, -85.])
    np.testing.assert_allclose(
        radar.elevation['data'][0:5],
        [2., 2., 2., 2., 2.])
    np.testing.assert_allclose(
        radar.range['data'][0:5],
        [14.98962308, 44.96886923, 74.94811538,
         104.92736153, 134.90660768])
    gate_lat = radar.gate_latitude['data'][0, 0:5]
    gate_lon = radar.gate_longitude['data'][0, 0:5]
    gate_alt = radar.gate_altitude['data'][0, 0:5]
    np.testing.assert_allclose(
        gate_lat, [38.95293483, 38.95291135, 38.95288786,
                   38.95286437, 38.95284089])
    np.testing.assert_allclose(
        gate_lon, [-76.8363515, -76.83669666, -76.83704182,
                   -76.83738699, -76.83773215])
    np.testing.assert_allclose(
        gate_alt, [62.84009906, 63.8864653, 64.93293721,
                   65.9795148, 67.02619806])
    obj.close()
    del radar


def test_add_solar_variable():
    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_NAV)
    new_obj = act.utils.geo_utils.add_solar_variable(obj)

    assert 'sun_variable' in list(new_obj.keys())
    assert new_obj['sun_variable'].values[10] == 1
    assert np.sum(new_obj['sun_variable'].values) >= 598

    new_obj = act.utils.geo_utils.add_solar_variable(obj, dawn_dusk=True)
    assert 'sun_variable' in list(new_obj.keys())
    assert new_obj['sun_variable'].values[10] == 1
    assert np.sum(new_obj['sun_variable'].values) >= 1234

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET1)
    new_obj = act.utils.geo_utils.add_solar_variable(obj, dawn_dusk=True)
    assert np.sum(new_obj['sun_variable'].values) >= 1046

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_IRTSST)
    obj = obj.fillna(0)
    new_obj = act.utils.geo_utils.add_solar_variable(obj)
    assert np.sum(new_obj['sun_variable'].values) >= 12

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_IRTSST)
    obj.drop_vars('lat')
    pytest.raises(
        ValueError, act.utils.geo_utils.add_solar_variable, obj)

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_IRTSST)
    obj.drop_vars('lon')
    pytest.raises(
        ValueError, act.utils.geo_utils.add_solar_variable, obj)
    obj.close()
    new_obj.close()


def test_reduce_time_ranges():
    time = pd.date_range(start='2020-01-01T00:00:00', freq='1min', periods=100)
    time = time.to_list()
    time = time[0:50] + time[60:]
    result = act.utils.datetime_utils.reduce_time_ranges(time)
    assert len(result) == 2
    assert result[1][1].minute == 39

    result = act.utils.datetime_utils.reduce_time_ranges(time, broken_barh=True)
    assert len(result) == 2


def test_planck_converter():
    wnum = 1100
    temp = 300
    radiance = 81.5
    result = act.utils.radiance_utils.planck_converter(wnum=wnum, temperature=temp)
    np.testing.assert_almost_equal(result, radiance, decimal=1)
    result = act.utils.radiance_utils.planck_converter(wnum=wnum, radiance=radiance)
    assert np.ceil(result) == temp
    np.testing.assert_raises(ValueError, act.utils.radiance_utils.planck_converter)


def test_solar_azimuth_elevation():

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_NAV)

    elevation, azimuth, distance = act.utils.geo_utils.get_solar_azimuth_elevation(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0], time=obj['time'].values,
        library='skyfield', temperature_C='standard', pressure_mbar='standard')
    assert np.isclose(np.nanmean(elevation), 26.408, atol=0.001)
    assert np.isclose(np.nanmean(azimuth), 179.732, atol=0.001)
    assert np.isclose(np.nanmean(distance), 0.985, atol=0.001)


def test_get_sunrise_sunset_noon():

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_NAV)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0],
        date=obj['time'].values[0], library='skyfield')
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0],
        date=obj['time'].values[0], library='skyfield', timezone=True)
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32, tzinfo=pytz.UTC)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4, tzinfo=pytz.UTC)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10, tzinfo=pytz.UTC)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0],
        date='20180201', library='skyfield')
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0],
        date=['20180201'], library='skyfield')
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0],
        date=datetime(2018, 2, 1), library='skyfield')
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0],
        date=datetime(2018, 2, 1, tzinfo=pytz.UTC), library='skyfield')
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=obj['lat'].values[0], longitude=obj['lon'].values[0],
        date=[datetime(2018, 2, 1)], library='skyfield')
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=85.0, longitude=-140., date=[datetime(2018, 6, 1)], library='skyfield')
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 3, 30, 10, 48, 48)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 9, 12, 8, 50, 14)
    assert noon[0].replace(microsecond=0) == datetime(2018, 6, 1, 21, 17, 52)


def test_is_sun_visible():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    result = act.utils.geo_utils.is_sun_visible(
        latitude=obj['lat'].values, longitude=obj['lon'].values,
        date_time=obj['time'].values)
    assert len(result) == 48
    assert sum(result) == 20

    result = act.utils.geo_utils.is_sun_visible(
        latitude=obj['lat'].values, longitude=obj['lon'].values,
        date_time=obj['time'].values[0])
    assert result == [False]

    result = act.utils.geo_utils.is_sun_visible(
        latitude=obj['lat'].values, longitude=obj['lon'].values,
        date_time=[datetime(2019, 11, 25, 13, 30, 00)])
    assert result == [True]

    result = act.utils.geo_utils.is_sun_visible(
        latitude=obj['lat'].values, longitude=obj['lon'].values,
        date_time=datetime(2019, 11, 25, 13, 30, 00))
    assert result == [True]


def test_convert_to_potential_temp():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    temp_var_name = 'temp_mean'
    press_var_name = 'atmos_pressure'
    temp = act.utils.data_utils.convert_to_potential_temp(
        obj, temp_var_name, press_var_name=press_var_name)
    assert np.isclose(np.nansum(temp), -4240.092, rtol=0.001, atol=0.001)
    temp = act.utils.data_utils.convert_to_potential_temp(
        temperature=obj[temp_var_name].values, pressure=obj[press_var_name].values,
        temp_var_units=obj[temp_var_name].attrs['units'],
        press_var_units=obj[press_var_name].attrs['units'])
    assert np.isclose(np.nansum(temp), -4240.092, rtol=0.001, atol=0.0011)

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.convert_to_potential_temp(
            temperature=obj[temp_var_name].values, pressure=obj[press_var_name].values,
            temp_var_units=obj[temp_var_name].attrs['units'])

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.convert_to_potential_temp(
            temperature=obj[temp_var_name].values, pressure=obj[press_var_name].values,
            press_var_units=obj[press_var_name].attrs['units'])


def test_height_adjusted_temperature():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    temp_var_name = 'temp_mean'
    press_var_name = 'atmos_pressure'
    temp = act.utils.data_utils.height_adjusted_temperature(
        obj, temp_var_name, height_difference=100, height_units='m',
        press_var_name=press_var_name)
    assert np.isclose(np.nansum(temp), -6834.291, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_temperature(
        obj, temp_var_name=temp_var_name, height_difference=-900,
        height_units='feet')
    assert np.isclose(np.nansum(temp), -1904.7257, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_temperature(
        obj, temp_var_name, height_difference=-200,
        height_units='m', press_var_name=press_var_name,
        pressure=102.325, press_var_units='kPa')
    assert np.isclose(np.nansum(temp), -2871.5435, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_temperature(
        height_difference=25.2, height_units='m',
        temperature=obj[temp_var_name].values,
        temp_var_units=obj[temp_var_name].attrs['units'],
        pressure=obj[press_var_name].values,
        press_var_units=obj[press_var_name].attrs['units'])
    assert np.isclose(np.nansum(temp), -5847.511, rtol=0.001, atol=0.001)

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.height_adjusted_temperature(
            height_difference=25.2, height_units='m',
            temperature=obj[temp_var_name].values,
            temp_var_units=None,
            pressure=obj[press_var_name].values,
            press_var_units=obj[press_var_name].attrs['units'])


def test_height_adjusted_pressure():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    press_var_name = 'atmos_pressure'
    temp = act.utils.data_utils.height_adjusted_pressure(
        obj=obj, press_var_name=press_var_name, height_difference=20,
        height_units='m')
    assert np.isclose(np.nansum(temp), 142020.83, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_pressure(
        height_difference=-100, height_units='ft',
        pressure=obj[press_var_name].values,
        press_var_units=obj[press_var_name].attrs['units'])
    assert np.isclose(np.nansum(temp), 142877.69, rtol=0.001, atol=0.001)

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.height_adjusted_pressure(
            height_difference=-100, height_units='ft',
            pressure=obj[press_var_name].values,
            press_var_units=None)
