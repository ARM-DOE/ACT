import importlib

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal
from contextlib import redirect_stdout
from io import StringIO

import act
from act.utils.data_utils import DatastreamParserARM as DatastreamParser

spec = importlib.util.find_spec('pyart')
if spec is not None:
    PYART_AVAILABLE = True
else:
    PYART_AVAILABLE = False


def test_add_in_nan():
    # Make a 1D array of 10 minute data
    time = np.arange('2019-01-01T01:00', '2019-01-01T01:10', dtype='datetime64[m]')
    time = time.astype('datetime64[us]')
    time = np.delete(time, range(3, 8))
    data = np.linspace(0.0, 8.0, time.size)

    time_filled, data_filled = act.utils.add_in_nan(xr.DataArray(time), xr.DataArray(data))
    assert isinstance(time_filled, xr.core.dataarray.DataArray)
    assert isinstance(data_filled, xr.core.dataarray.DataArray)

    time_filled, data_filled = act.utils.add_in_nan(time, data)
    assert isinstance(time_filled, np.ndarray)
    assert isinstance(data_filled, np.ndarray)

    assert time_filled[3] == np.datetime64('2019-01-01T01:05:00')
    assert time_filled[4] == np.datetime64('2019-01-01T01:08:00')
    assert np.isnan(data_filled[3])
    assert data_filled[4] == 6.0

    time_filled, data_filled = act.utils.add_in_nan(time[0], data[0])
    assert time_filled == time[0]
    assert data_filled == data[0]

    # Check for multiple instances of missing data periods
    time = np.arange('2019-01-01T01:00', '2019-01-01T02:00', dtype='datetime64[m]')
    time = np.delete(time, range(3, 8))
    time = np.delete(time, range(33, 36))
    data = np.linspace(0.0, 10.0, time.size)

    time_filled, data_filled = act.utils.add_in_nan(time, data)
    assert time_filled.size == 54
    assert data_filled.size == 54
    index = np.where(time_filled == np.datetime64('2019-01-01T01:37'))[0]
    assert index[0] == 33
    assert np.isclose(data_filled[33], 6.27450)
    index = np.where(time_filled == np.datetime64('2019-01-01T01:38'))[0]
    assert index.size == 0
    index = np.where(time_filled == np.datetime64('2019-01-01T01:39'))[0]
    assert index[0] == 34
    assert np.isnan(data_filled[34])
    index = np.where(time_filled == np.datetime64('2019-01-01T01:40'))[0]
    assert index.size == 0

    # Test for 2D data
    time = np.arange('2019-01-01T01:00', '2019-01-01T02:00', dtype='datetime64[m]')
    data = np.random.random((len(time), 25))

    time = np.delete(time, range(3, 8))
    data = np.delete(data, range(3, 8), axis=0)
    time_filled, data_filled = act.utils.add_in_nan(time, data)

    assert np.count_nonzero(np.isnan(data_filled[3, :])) == 25
    assert len(time_filled) == len(time) + 2


def test_get_missing_value():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    missing = act.utils.data_utils.get_missing_value(
        ds, 'latent_heat_flux', use_FillValue=True, add_if_missing_in_ds=True
    )
    assert missing == -9999

    ds['latent_heat_flux'].attrs['missing_value'] = -9998
    missing = act.utils.data_utils.get_missing_value(ds, 'latent_heat_flux')
    assert missing == -9998


def test_convert_units():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    data = ds['soil_temp_1'].values
    in_units = ds['soil_temp_1'].attrs['units']
    r_data = act.utils.data_utils.convert_units(data, in_units, 'K')
    assert np.ceil(r_data[0]) == 285

    data = act.utils.data_utils.convert_units(r_data, 'K', 'C')
    assert np.ceil(data[0]) == 12

    with np.testing.assert_raises(ValueError):
        ds.utils.change_units()

    desired_unit = 'degF'
    skip_vars = [ii for ii in ds.data_vars if ii.startswith('qc_')]
    ds.utils.change_units(
        variables=None,
        desired_unit=desired_unit,
        skip_variables=skip_vars,
        skip_standard=True,
    )
    units = []
    for var_name in ds.data_vars:
        try:
            units.append(ds[var_name].attrs['units'])
        except KeyError:
            pass
    indices = [i for i, x in enumerate(units) if x == desired_unit]
    assert indices == [0, 2, 4, 6, 8, 32, 34, 36, 38, 40]

    var_name = 'home_signal_15'
    desired_unit = 'V'
    ds.utils.change_units(var_name, desired_unit, skip_variables='lat')
    assert ds[var_name].attrs['units'] == desired_unit

    var_names = ['home_signal_15', 'home_signal_30']
    ds.utils.change_units(var_names, desired_unit)
    for var_name in var_names:
        assert ds[var_name].attrs['units'] == desired_unit

    ds.close()
    del ds

    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_CEIL1)
    var_name = 'range'
    desired_unit = 'km'
    ds = ds.utils.change_units(var_name, desired_unit)
    assert ds[var_name].attrs['units'] == desired_unit
    assert np.isclose(np.sum(ds[var_name].values), 952.56, atol=0.01)

    ds.close()
    del ds

    # Test if exception or print statement is issued when an error occurs with units string
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    with np.testing.assert_raises(ValueError):
        ds.utils.change_units('home_signal_15', 'not_a_real_unit_string', raise_error=True)

    with np.testing.assert_raises(ValueError):
        ds.utils.change_units('not_a_real_variable_name', 'degC', raise_error=True)

    f = StringIO()
    var_name = 'home_signal_15'
    unit = 'not_a_real_unit_string'
    with redirect_stdout(f):
        ds.utils.change_units('home_signal_15', 'not_a_real_unit_string', verbose=True)
    s = f.getvalue()
    assert (
        s.strip()
        == f"Unable to convert '{var_name}' to units of '{unit}'. Skipping unit converstion for '{var_name}'."
    )
    ds.close()
    del ds


def test_ts_weighted_average():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MET_WILDCARD)
    cf_ds = {
        'sgpmetE13.b1': {
            'variable': [
                'tbrg_precip_total',
                'org_precip_rate_mean',
                'pwd_precip_rate_mean_1min',
            ],
            'weight': [0.8, 0.15, 0.05],
            'ds': ds,
        }
    }
    data = act.utils.data_utils.ts_weighted_average(cf_ds)

    np.testing.assert_almost_equal(np.sum(data), 84.9, decimal=1)


def test_accum_precip():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MET_WILDCARD)

    ds = act.utils.accumulate_precip(ds, 'tbrg_precip_total')
    dmax = round(np.nanmax(ds['tbrg_precip_total_accumulated']))
    assert np.isclose(dmax, 13.0, atol=0.01)

    ds = act.utils.accumulate_precip(ds, 'tbrg_precip_total', time_delta=60)
    dmax = round(np.nanmax(ds['tbrg_precip_total_accumulated']))
    assert np.isclose(dmax, 13.0, atol=0.01)

    ds['tbrg_precip_total'].attrs['units'] = 'mm/hr'
    ds = act.utils.accumulate_precip(ds, 'tbrg_precip_total')
    dmax = np.round(np.nanmax(ds['tbrg_precip_total_accumulated']), 2)
    assert np.isclose(dmax, 0.22, atol=0.01)


@pytest.mark.skipif(not PYART_AVAILABLE, reason='Py-ART is not installed.')
def test_create_pyart_obj():
    try:
        ds = act.io.mpl.read_sigma_mplv5(act.tests.EXAMPLE_SIGMA_MPLV5)
    except Exception:
        return

    radar = act.utils.create_pyart_obj(ds, range_var='range')
    variables = list(radar.fields)
    assert 'nrb_copol' in variables
    assert 'nrb_crosspol' in variables
    assert radar.sweep_start_ray_index['data'][-1] == 67
    assert radar.sweep_end_ray_index['data'][-1] == 101
    assert radar.fixed_angle['data'] == 2.0
    assert radar.scan_type == 'ppi'
    assert radar.sweep_mode['data'] == 'ppi'
    np.testing.assert_allclose(radar.sweep_number['data'][-3:], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(radar.sweep_number['data'][0:3], [0.0, 0.0, 0.0])

    # coordinates
    np.testing.assert_allclose(radar.azimuth['data'][0:5], [-95.0, -92.5, -90.0, -87.5, -85.0])
    np.testing.assert_allclose(radar.elevation['data'][0:5], [2.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(
        radar.range['data'][0:5],
        [14.98962308, 44.96886923, 74.94811538, 104.92736153, 134.90660768],
    )
    gate_lat = radar.gate_latitude['data'][0, 0:5]
    gate_lon = radar.gate_longitude['data'][0, 0:5]
    gate_alt = radar.gate_altitude['data'][0, 0:5]
    np.testing.assert_allclose(
        gate_lat, [38.95293483, 38.95291135, 38.95288786, 38.95286437, 38.95284089]
    )
    np.testing.assert_allclose(
        gate_lon, [-76.8363515, -76.83669666, -76.83704182, -76.83738699, -76.83773215]
    )
    np.testing.assert_allclose(
        gate_alt, [62.84009906, 63.8864653, 64.93293721, 65.9795148, 67.02619806]
    )
    ds.close()
    del radar


def test_convert_to_potential_temp():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    temp_var_name = 'temp_mean'
    press_var_name = 'atmos_pressure'
    temp = act.utils.data_utils.convert_to_potential_temp(
        ds, temp_var_name, press_var_name=press_var_name
    )
    assert np.isclose(np.nansum(temp), -4240.092, rtol=0.001, atol=0.001)
    temp = act.utils.data_utils.convert_to_potential_temp(
        temperature=ds[temp_var_name].values,
        pressure=ds[press_var_name].values,
        temp_var_units=ds[temp_var_name].attrs['units'],
        press_var_units=ds[press_var_name].attrs['units'],
    )
    assert np.isclose(np.nansum(temp), -4240.092, rtol=0.001, atol=0.0011)

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.convert_to_potential_temp(
            temperature=ds[temp_var_name].values,
            pressure=ds[press_var_name].values,
            temp_var_units=ds[temp_var_name].attrs['units'],
        )

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.convert_to_potential_temp(
            temperature=ds[temp_var_name].values,
            pressure=ds[press_var_name].values,
            press_var_units=ds[press_var_name].attrs['units'],
        )


def test_height_adjusted_temperature():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    temp_var_name = 'temp_mean'
    press_var_name = 'atmos_pressure'
    temp = act.utils.data_utils.height_adjusted_temperature(
        ds,
        temp_var_name,
        height_difference=100,
        height_units='m',
        press_var_name=press_var_name,
    )
    assert np.isclose(np.nansum(temp), -6834.291, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_temperature(
        ds, temp_var_name=temp_var_name, height_difference=-900, height_units='feet'
    )
    assert np.isclose(np.nansum(temp), -1904.7257, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_temperature(
        ds,
        temp_var_name,
        height_difference=-200,
        height_units='m',
        press_var_name=press_var_name,
        pressure=102.325,
        press_var_units='kPa',
    )
    assert np.isclose(np.nansum(temp), -2871.5435, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_temperature(
        height_difference=25.2,
        height_units='m',
        temperature=ds[temp_var_name].values,
        temp_var_units=ds[temp_var_name].attrs['units'],
        pressure=ds[press_var_name].values,
        press_var_units=ds[press_var_name].attrs['units'],
    )
    assert np.isclose(np.nansum(temp), -5847.511, rtol=0.001, atol=0.001)

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.height_adjusted_temperature(
            height_difference=25.2,
            height_units='m',
            temperature=ds[temp_var_name].values,
            temp_var_units=None,
            pressure=ds[press_var_name].values,
            press_var_units=ds[press_var_name].attrs['units'],
        )


def test_height_adjusted_pressure():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MET1)

    press_var_name = 'atmos_pressure'
    temp = act.utils.data_utils.height_adjusted_pressure(
        ds=ds, press_var_name=press_var_name, height_difference=20, height_units='m'
    )
    assert np.isclose(np.nansum(temp), 142020.83, rtol=0.001, atol=0.001)

    temp = act.utils.data_utils.height_adjusted_pressure(
        height_difference=-100,
        height_units='ft',
        pressure=ds[press_var_name].values,
        press_var_units=ds[press_var_name].attrs['units'],
    )
    assert np.isclose(np.nansum(temp), 142877.69, rtol=0.001, atol=0.001)

    with np.testing.assert_raises(ValueError):
        temp = act.utils.data_utils.height_adjusted_pressure(
            height_difference=-100,
            height_units='ft',
            pressure=ds[press_var_name].values,
            press_var_units=None,
        )


def test_datastreamparser():
    pytest.raises(ValueError, DatastreamParser, 123)

    fn_obj = DatastreamParser()
    pytest.raises(ValueError, fn_obj.set_datastream, None)

    fn_obj = DatastreamParser()
    assert fn_obj.site is None
    assert fn_obj.datastream_class is None
    assert fn_obj.facility is None
    assert fn_obj.level is None
    assert fn_obj.datastream is None
    assert fn_obj.date is None
    assert fn_obj.time is None
    assert fn_obj.ext is None
    del fn_obj

    fn_obj = DatastreamParser('/data/sgp/sgpmetE13.b1/sgpmetE13.b1.20190501.024254.nc')
    assert fn_obj.site == 'sgp'
    assert fn_obj.datastream_class == 'met'
    assert fn_obj.facility == 'E13'
    assert fn_obj.level == 'b1'
    assert fn_obj.datastream == 'sgpmetE13.b1'
    assert fn_obj.date == '20190501'
    assert fn_obj.time == '024254'
    assert fn_obj.ext == 'nc'

    fn_obj.set_datastream('nsatwrC1.a0.19991230.233451.cdf')
    assert fn_obj.site == 'nsa'
    assert fn_obj.datastream_class == 'twr'
    assert fn_obj.facility == 'C1'
    assert fn_obj.level == 'a0'
    assert fn_obj.datastream == 'nsatwrC1.a0'
    assert fn_obj.date == '19991230'
    assert fn_obj.time == '233451'
    assert fn_obj.ext == 'cdf'

    fn_obj = DatastreamParser('nsaitscomplicatedX1.00.991230.2334.txt')
    assert fn_obj.site == 'nsa'
    assert fn_obj.datastream_class == 'itscomplicated'
    assert fn_obj.facility == 'X1'
    assert fn_obj.level == '00'
    assert fn_obj.datastream == 'nsaitscomplicatedX1.00'
    assert fn_obj.date == '991230'
    assert fn_obj.time == '2334'
    assert fn_obj.ext == 'txt'

    fn_obj = DatastreamParser('sgpmetE13.b1')
    assert fn_obj.site == 'sgp'
    assert fn_obj.datastream_class == 'met'
    assert fn_obj.facility == 'E13'
    assert fn_obj.level == 'b1'
    assert fn_obj.datastream == 'sgpmetE13.b1'
    assert fn_obj.date is None
    assert fn_obj.time is None
    assert fn_obj.ext is None

    fn_obj = DatastreamParser('sgpmetE13')
    assert fn_obj.site == 'sgp'
    assert fn_obj.datastream_class == 'met'
    assert fn_obj.facility == 'E13'
    assert fn_obj.level is None
    assert fn_obj.datastream is None
    assert fn_obj.date is None
    assert fn_obj.time is None
    assert fn_obj.ext is None

    fn_obj = DatastreamParser('sgpmet')
    assert fn_obj.site == 'sgp'
    assert fn_obj.datastream_class == 'met'
    assert fn_obj.facility is None
    assert fn_obj.level is None
    assert fn_obj.datastream is None
    assert fn_obj.date is None
    assert fn_obj.time is None
    assert fn_obj.ext is None

    fn_obj = DatastreamParser('sgp')
    assert fn_obj.site == 'sgp'
    assert fn_obj.datastream_class is None
    assert fn_obj.facility is None
    assert fn_obj.level is None
    assert fn_obj.datastream is None
    assert fn_obj.date is None
    assert fn_obj.time is None
    assert fn_obj.ext is None

    fn_obj = DatastreamParser('sg')
    assert fn_obj.site is None
    assert fn_obj.datastream_class is None
    assert fn_obj.facility is None
    assert fn_obj.level is None
    assert fn_obj.datastream is None
    assert fn_obj.date is None
    assert fn_obj.time is None
    assert fn_obj.ext is None
    del fn_obj


def test_arm_site_location_search():
    # Test for many facilities
    test_dict_many = act.utils.arm_site_location_search(site_code='sgp', facility_code=None)
    assert len(test_dict_many) > 30
    assert list(test_dict_many)[0] == 'sgp A1'
    assert list(test_dict_many)[2] == 'sgp A3'

    assert_almost_equal(test_dict_many[list(test_dict_many)[0]]['latitude'], 37.843058)
    assert_almost_equal(test_dict_many[list(test_dict_many)[0]]['longitude'], -97.020569)
    assert_almost_equal(test_dict_many[list(test_dict_many)[2]]['latitude'], 37.626)
    assert_almost_equal(test_dict_many[list(test_dict_many)[2]]['longitude'], -96.882)

    # Test for one facility
    test_dict_one = act.utils.arm_site_location_search(site_code='sgp', facility_code='I5')
    assert len(test_dict_one) == 1
    assert list(test_dict_one)[0] == 'sgp I5'
    assert_almost_equal(test_dict_one[list(test_dict_one)[0]]['latitude'], 36.491178)
    assert_almost_equal(test_dict_one[list(test_dict_one)[0]]['longitude'], -97.593936)

    # Test for a facility with no latitude and longitude information
    test_dict_no_coord = act.utils.arm_site_location_search(site_code='sgp', facility_code='A6')
    assert list(test_dict_no_coord)[0] == 'sgp A6'
    assert test_dict_no_coord[list(test_dict_no_coord)[0]]['latitude'] is None
    assert test_dict_no_coord[list(test_dict_no_coord)[0]]['longitude'] is None

    # Test for another site
    test_dict_nsa = act.utils.arm_site_location_search(site_code='nsa', facility_code=None)
    assert len(test_dict_nsa) > 5
    assert list(test_dict_nsa)[0] == 'nsa C1'
    assert test_dict_nsa[list(test_dict_nsa)[0]]['latitude'] == 71.323
    assert test_dict_nsa[list(test_dict_nsa)[0]]['longitude'] == -156.615
