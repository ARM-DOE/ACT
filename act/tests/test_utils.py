import act
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime


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


def test_get_missing_value():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    missing = act.utils.data_utils.get_missing_value(obj, 'lv_e', use_FillValue=True)
    assert missing == -9999


def test_convert_units():
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    data = obj['soil_temp_1'].values
    in_units = obj['soil_temp_1'].attrs['units']
    r_data = act.utils.data_utils.convert_units(data, in_units, 'K')
    assert np.ceil(r_data[0]) == 285

    data = act.utils.data_utils.convert_units(r_data, 'K', 'C')
    assert np.ceil(data[0]) == 12


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


def test_add_solar_variable():
    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_NAV)
    new_obj = act.utils.geo_utils.add_solar_variable(obj)

    assert 'sun_variable' in list(new_obj.keys())
    assert new_obj['sun_variable'].values[10] == 1
    assert np.sum(new_obj['sun_variable'].values) >= 519

    new_obj = act.utils.geo_utils.add_solar_variable(obj, dawn_dusk=True)
    assert 'sun_variable' in list(new_obj.keys())
    assert new_obj['sun_variable'].values[10] == 1
    assert np.sum(new_obj['sun_variable'].values) >= 519

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET1)
    new_obj = act.utils.geo_utils.add_solar_variable(obj, dawn_dusk=True)
    assert np.sum(new_obj['sun_variable'].values) >= 582

    obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_IRTSST)
    obj = obj.fillna(0)
    new_obj = act.utils.geo_utils.add_solar_variable(obj)
    assert np.sum(new_obj['sun_variable'].values) >= 12


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
