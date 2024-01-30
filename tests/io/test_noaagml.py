import numpy as np

import act
from act.io import read_gml
from act.tests import sample_files


def test_read_gml():
    # Test Radiation
    ds = read_gml(sample_files.EXAMPLE_GML_RADIATION, datatype='RADIATION')
    assert np.isclose(np.nansum(ds['solar_zenith_angle']), 1725.28)
    assert np.isclose(np.nansum(ds['upwelling_infrared_case_temp']), 4431.88)
    assert (
        ds['upwelling_infrared_case_temp'].attrs['ancillary_variables']
        == 'qc_upwelling_infrared_case_temp'
    )
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_values'] == [0, 1, 2]
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_meanings'] == [
        'Not failing any tests',
        'Knowingly bad value',
        'Should be used with scrutiny',
    ]
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_assessments'] == [
        'Good',
        'Bad',
        'Indeterminate',
    ]
    assert ds['time'].values[-1] == np.datetime64('2021-01-01T00:17:00')

    ds = read_gml(sample_files.EXAMPLE_GML_RADIATION, convert_missing=False)
    assert np.isclose(np.nansum(ds['solar_zenith_angle']), 1725.28)
    assert np.isclose(np.nansum(ds['upwelling_infrared_case_temp']), 4431.88)
    assert (
        ds['upwelling_infrared_case_temp'].attrs['ancillary_variables']
        == 'qc_upwelling_infrared_case_temp'
    )
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_values'] == [0, 1, 2]
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_meanings'] == [
        'Not failing any tests',
        'Knowingly bad value',
        'Should be used with scrutiny',
    ]
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_assessments'] == [
        'Good',
        'Bad',
        'Indeterminate',
    ]
    assert ds['time'].values[-1] == np.datetime64('2021-01-01T00:17:00')

    # Test MET
    ds = read_gml(sample_files.EXAMPLE_GML_MET, datatype='MET')
    assert np.isclose(np.nansum(ds['wind_speed'].values), 148.1)
    assert ds['wind_speed'].attrs['units'] == 'm/s'
    assert np.isnan(ds['wind_speed'].attrs['_FillValue'])
    assert np.sum(np.isnan(ds['preciptation_intensity'].values)) == 20
    assert ds['preciptation_intensity'].attrs['units'] == 'mm/hour'
    assert ds['time'].values[0] == np.datetime64('2020-01-01T00:00:00')

    ds = read_gml(sample_files.EXAMPLE_GML_MET, convert_missing=False)
    assert np.isclose(np.nansum(ds['wind_speed'].values), 148.1)
    assert ds['wind_speed'].attrs['units'] == 'm/s'
    assert np.isclose(ds['wind_speed'].attrs['_FillValue'], -999.9)
    assert np.sum(ds['preciptation_intensity'].values) == -1980
    assert ds['preciptation_intensity'].attrs['units'] == 'mm/hour'
    assert ds['time'].values[0] == np.datetime64('2020-01-01T00:00:00')

    # Test Ozone
    ds = read_gml(sample_files.EXAMPLE_GML_OZONE, datatype='OZONE')
    assert np.isclose(np.nansum(ds['ozone'].values), 582.76)
    assert ds['ozone'].attrs['long_name'] == 'Ozone'
    assert ds['ozone'].attrs['units'] == 'ppb'
    assert np.isnan(ds['ozone'].attrs['_FillValue'])
    assert ds['time'].values[0] == np.datetime64('2020-12-01T00:00:00')

    ds = read_gml(sample_files.EXAMPLE_GML_OZONE)
    assert np.isclose(np.nansum(ds['ozone'].values), 582.76)
    assert ds['ozone'].attrs['long_name'] == 'Ozone'
    assert ds['ozone'].attrs['units'] == 'ppb'
    assert np.isnan(ds['ozone'].attrs['_FillValue'])
    assert ds['time'].values[0] == np.datetime64('2020-12-01T00:00:00')

    # Test Carbon Dioxide
    ds = read_gml(sample_files.EXAMPLE_GML_CO2, datatype='co2')
    assert np.isclose(np.nansum(ds['co2'].values), 2307.630)
    assert (
        ds['qc_co2'].values == np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    ).all()
    assert ds['co2'].attrs['units'] == 'ppm'
    assert np.isnan(ds['co2'].attrs['_FillValue'])
    assert ds['qc_co2'].attrs['flag_assessments'] == ['Bad', 'Indeterminate']
    assert ds['latitude'].attrs['standard_name'] == 'latitude'

    ds = read_gml(sample_files.EXAMPLE_GML_CO2, convert_missing=False)
    assert np.isclose(np.nansum(ds['co2'].values), -3692.3098)
    assert ds['co2'].attrs['_FillValue'] == -999.99
    assert (
        ds['qc_co2'].values == np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    ).all()
    assert ds['co2'].attrs['units'] == 'ppm'
    assert np.isclose(ds['co2'].attrs['_FillValue'], -999.99)
    assert ds['qc_co2'].attrs['flag_assessments'] == ['Bad', 'Indeterminate']
    assert ds['latitude'].attrs['standard_name'] == 'latitude'

    # Test Halocarbon
    ds = read_gml(sample_files.EXAMPLE_GML_HALO, datatype='HALO')
    assert np.isclose(np.nansum(ds['CCl4'].values), 1342.65)
    assert ds['CCl4'].attrs['units'] == 'ppt'
    assert ds['CCl4'].attrs['long_name'] == 'Carbon Tetrachloride (CCl4) daily median'
    assert np.isnan(ds['CCl4'].attrs['_FillValue'])
    assert ds['time'].values[0] == np.datetime64('1998-06-16T00:00:00')

    ds = read_gml(sample_files.EXAMPLE_GML_HALO)
    assert np.isclose(np.nansum(ds['CCl4'].values), 1342.65)
    assert ds['CCl4'].attrs['units'] == 'ppt'
    assert ds['CCl4'].attrs['long_name'] == 'Carbon Tetrachloride (CCl4) daily median'
    assert np.isnan(ds['CCl4'].attrs['_FillValue'])
    assert ds['time'].values[0] == np.datetime64('1998-06-16T00:00:00')


def test_read_surfrad():
    url = ['https://gml.noaa.gov/aftp/data/radiation/surfrad/Boulder_CO/2023/tbl23008.dat']
    ds = act.io.noaagml.read_surfrad(url)

    assert 'qc_pressure' in ds
    assert 'time' in ds
    assert ds['wind_speed'].attrs['units'] == 'ms^-1'
    assert len(ds) == 48
    assert ds['temperature'].values[0] == 2.0
    assert 'standard_name' in ds['temperature'].attrs
    assert ds['temperature'].attrs['standard_name'] == 'air_temperature'
