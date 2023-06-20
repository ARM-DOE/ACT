import glob
from os import PathLike
from pathlib import Path
import random
from string import ascii_letters
import tempfile

import fsspec
import numpy as np
import pytest

import act
import act.tests.sample_files as sample_files
from act.io import read_gml, read_psl_wind_profiler_temperature, icartt
from act.io.noaapsl import read_psl_surface_met


def test_io():
    ds = act.io.armfiles.read_netcdf([act.tests.EXAMPLE_MET1])
    assert 'temp_mean' in ds.variables.keys()
    assert 'rh_mean' in ds.variables.keys()
    assert ds.attrs['_arm_standards_flag'] == (1 << 0)

    with np.testing.assert_raises(OSError):
        ds = act.io.armfiles.read_netcdf([])

    ds = act.io.armfiles.read_netcdf([], return_None=True)
    assert ds is None
    ds = act.io.armfiles.read_netcdf(['./randomfile.nc'], return_None=True)
    assert ds is None

    ds = act.io.armfiles.read_netcdf([act.tests.EXAMPLE_MET_TEST1])
    assert 'time' in ds

    ds = act.io.armfiles.read_netcdf([act.tests.EXAMPLE_MET_TEST2])
    assert ds['time'].values[10].astype('datetime64[ms]') == np.datetime64('2019-01-01T00:10:00', 'ms')

    ds = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET1, use_base_time=True, drop_variables='time')
    assert 'time' in ds
    assert np.issubdtype(ds['time'].dtype, np.datetime64)
    assert ds['time'].values[10].astype('datetime64[ms]') == np.datetime64('2019-01-01T00:10:00', 'ms')

    del ds


def test_keep_variables():

    var_names = [
        'temp_mean',
        'rh_mean',
        'wdir_vec_mean',
        'tbrg_precip_total_corr',
        'atmos_pressure',
        'wspd_vec_mean',
        'pwd_pw_code_inst',
        'pwd_pw_code_15min',
        'pwd_mean_vis_10min',
        'logger_temp',
        'pwd_precip_rate_mean_1min',
        'pwd_cumul_snow',
        'pwd_mean_vis_1min',
        'pwd_pw_code_1hr',
        'org_precip_rate_mean',
        'tbrg_precip_total',
        'pwd_cumul_rain',
    ]
    var_names = var_names + ['qc_' + ii for ii in var_names]
    drop_variables = act.io.armfiles.keep_variables_to_drop_variables(
        act.tests.EXAMPLE_MET1, var_names
    )

    expected_drop_variables = [
        'wdir_vec_std',
        'base_time',
        'alt',
        'qc_wspd_arith_mean',
        'pwd_err_code',
        'logger_volt',
        'temp_std',
        'lon',
        'qc_logger_volt',
        'time_offset',
        'wspd_arith_mean',
        'lat',
        'vapor_pressure_std',
        'vapor_pressure_mean',
        'rh_std',
        'qc_vapor_pressure_mean',
    ]
    assert drop_variables.sort() == expected_drop_variables.sort()

    ds = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET1, keep_variables='temp_mean')
    assert list(ds.data_vars) == ['temp_mean']
    del ds

    var_names = ['temp_mean', 'qc_temp_mean']
    ds = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET1, keep_variables=var_names, drop_variables='nonsense'
    )
    assert list(ds.data_vars).sort() == var_names.sort()
    del ds

    var_names = ['temp_mean', 'qc_temp_mean', 'alt', 'lat', 'lon']
    ds = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET_WILDCARD, keep_variables=var_names, drop_variables=['lon']
    )
    var_names = list(set(var_names) - {'lon'})
    assert list(ds.data_vars).sort() == var_names.sort()
    del ds

    filenames = Path(act.tests.EXAMPLE_MET_WILDCARD).parent
    filenames = list(filenames.glob(Path(act.tests.EXAMPLE_MET_WILDCARD).name))
    var_names = ['temp_mean', 'qc_temp_mean', 'alt', 'lat', 'lon']
    ds = act.io.armfiles.read_netcdf(filenames, keep_variables=var_names)
    assert list(ds.data_vars).sort() == var_names.sort()
    del ds


def test_io_mfdataset():
    met_ds = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET_WILDCARD)
    met_ds.load()
    assert 'temp_mean' in met_ds.variables.keys()
    assert 'rh_mean' in met_ds.variables.keys()
    assert len(met_ds.attrs['_file_times']) == 7
    assert met_ds.attrs['_arm_standards_flag'] == (1 << 0)
    met_ds.close()
    del met_ds

    met_ds = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET_WILDCARD, cleanup_qc=True)
    met_ds.load()
    var_name = 'temp_mean'
    qc_var_name = 'qc_' + var_name
    attr_names = [
        'long_name',
        'units',
        'flag_masks',
        'flag_meanings',
        'flag_assessments',
        'fail_min',
        'fail_max',
        'fail_delta',
        'standard_name',
    ]
    assert var_name in met_ds.variables.keys()
    assert qc_var_name in met_ds.variables.keys()
    assert sorted(attr_names) == sorted(list(met_ds[qc_var_name].attrs.keys()))
    assert met_ds[qc_var_name].attrs['flag_masks'] == [1, 2, 4, 8]
    assert met_ds[qc_var_name].attrs['flag_assessments'] == ['Bad', 'Bad', 'Bad', 'Indeterminate']
    met_ds.close()
    del met_ds


def test_io_csv():
    headers = [
        'day',
        'month',
        'year',
        'time',
        'pasquill',
        'wdir_60m',
        'wspd_60m',
        'wdir_60m_std',
        'temp_60m',
        'wdir_10m',
        'wspd_10m',
        'wdir_10m_std',
        'temp_10m',
        'temp_dp',
        'rh',
        'avg_temp_diff',
        'total_precip',
        'solar_rad',
        'net_rad',
        'atmos_press',
        'wv_pressure',
        'temp_soil_10cm',
        'temp_soil_100cm',
        'temp_soil_10ft',
    ]
    anl_ds = act.io.csvfiles.read_csv(act.tests.EXAMPLE_ANL_CSV, sep=r'\s+', column_names=headers)
    assert 'temp_60m' in anl_ds.variables.keys()
    assert 'rh' in anl_ds.variables.keys()
    assert anl_ds['temp_60m'].values[10] == -1.7
    anl_ds.close()

    files = glob.glob(act.tests.EXAMPLE_MET_CSV)
    ds = act.io.csvfiles.read_csv(files[0])
    assert 'date_time' in ds
    assert '_datastream' in ds.attrs


def test_io_dod():
    dims = {'time': 1440, 'drop_diameter': 50}

    try:
        ds = act.io.armfiles.create_ds_from_arm_dod(
            'vdis.b1', dims, version='1.2', scalar_fill_dim='time'
        )
        assert 'moment1' in ds
        assert len(ds['base_time'].values) == 1440
        assert len(ds['drop_diameter'].values) == 50
        with np.testing.assert_warns(UserWarning):
            ds2 = act.io.armfiles.create_ds_from_arm_dod('vdis.b1', dims, scalar_fill_dim='time')
        assert 'moment1' in ds2
        assert len(ds2['base_time'].values) == 1440
        assert len(ds2['drop_diameter'].values) == 50
        with np.testing.assert_raises(ValueError):
            ds = act.io.armfiles.create_ds_from_arm_dod('vdis.b1', {}, version='1.2')
        ds = act.io.armfiles.create_ds_from_arm_dod(
            sample_files.EXAMPLE_DOD, dims, version=1.2, scalar_fill_dim='time',
            local_file=True)
        assert 'moment1' in ds
        assert len(ds['base_time'].values) == 1440
        assert len(ds['drop_diameter'].values) == 50
    except Exception:
        return
    ds.close()
    ds2.close()


def test_io_write():
    sonde_ds = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_SONDE1)
    sonde_ds.clean.cleanup()

    with tempfile.TemporaryDirectory() as tmpdirname:
        write_file = Path(tmpdirname, Path(sample_files.EXAMPLE_SONDE1).name)
        keep_vars = ['tdry', 'qc_tdry', 'dp', 'qc_dp']
        for var_name in list(sonde_ds.data_vars):
            if var_name not in keep_vars:
                del sonde_ds[var_name]
        sonde_ds.write.write_netcdf(path=write_file, FillValue=-9999)

        sonde_ds_read = act.io.armfiles.read_netcdf(str(write_file))
        assert list(sonde_ds_read.data_vars) == keep_vars
        assert isinstance(sonde_ds_read['qc_tdry'].attrs['flag_meanings'], str)
        assert sonde_ds_read['qc_tdry'].attrs['flag_meanings'].count('__') == 21
        for attr in ['qc_standards_version', 'qc_method', 'qc_comment']:
            assert attr not in list(sonde_ds_read.attrs)
        sonde_ds_read.close()
        del sonde_ds_read

    sonde_ds.close()

    sonde_ds = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_EBBR1)
    sonde_ds.clean.cleanup()
    assert 'fail_min' in sonde_ds['qc_home_signal_15'].attrs
    assert 'standard_name' in sonde_ds['qc_home_signal_15'].attrs
    assert 'flag_masks' in sonde_ds['qc_home_signal_15'].attrs

    with tempfile.TemporaryDirectory() as tmpdirname:
        cf_convention = 'CF-1.8'
        write_file = Path(tmpdirname, Path(sample_files.EXAMPLE_EBBR1).name)
        sonde_ds.write.write_netcdf(
            path=write_file,
            make_copy=False,
            join_char='_',
            cf_compliant=True,
            cf_convention=cf_convention,
        )

        sonde_ds_read = act.io.armfiles.read_netcdf(str(write_file))

        assert cf_convention in sonde_ds_read.attrs['Conventions'].split()
        assert sonde_ds_read.attrs['FeatureType'] == 'timeSeries'
        global_att_keys = [ii for ii in sonde_ds_read.attrs.keys() if not ii.startswith('_')]
        assert global_att_keys[-1] == 'history'
        assert sonde_ds_read['alt'].attrs['axis'] == 'Z'
        assert sonde_ds_read['alt'].attrs['positive'] == 'up'

        sonde_ds_read.close()
        del sonde_ds_read

    sonde_ds.close()

    ds = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_CEIL1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cf_convention = 'CF-1.8'
        write_file = Path(tmpdirname, Path(sample_files.EXAMPLE_CEIL1).name)
        ds.write.write_netcdf(
            path=write_file,
            make_copy=False,
            join_char='_',
            cf_compliant=True,
            cf_convention=cf_convention,
        )

        ds_read = act.io.armfiles.read_netcdf(str(write_file))

        assert cf_convention in ds_read.attrs['Conventions'].split()
        assert ds_read.attrs['FeatureType'] == 'timeSeriesProfile'
        assert len(ds_read.dims) > 1

        ds_read.close()
        del ds_read


def test_clean_cf_qc():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_MET1, cleanup_qc=True)
        ds.load()
        var_name = 'temp_mean'
        qc_var_name = 'qc_' + var_name
        ds.qcfilter.remove_test(var_name, test_number=4)
        ds.qcfilter.remove_test(var_name, test_number=3)
        ds.qcfilter.remove_test(var_name, test_number=2)
        ds[qc_var_name].attrs['flag_masks'] = ds[qc_var_name].attrs['flag_masks'][0]
        flag_meanings = ds[qc_var_name].attrs['flag_meanings'][0]
        ds[qc_var_name].attrs['flag_meanings'] = flag_meanings.replace(' ', '__')
        flag_meanings = ds[qc_var_name].attrs['flag_assessments'][0]
        ds[qc_var_name].attrs['flag_assessments'] = flag_meanings.replace(' ', '__')

        write_file = str(Path(tmpdirname, Path(sample_files.EXAMPLE_MET1).name))
        ds.write.write_netcdf(path=write_file, cf_compliant=True)
        ds.close()
        del ds

        read_ds = act.io.armfiles.read_netcdf(write_file, cleanup_qc=True)
        read_ds.load()

        assert type(read_ds[qc_var_name].attrs['flag_masks']).__module__ == 'numpy'
        assert read_ds[qc_var_name].attrs['flag_masks'].size == 1
        assert read_ds[qc_var_name].attrs['flag_masks'][0] == 1
        assert isinstance(read_ds[qc_var_name].attrs['flag_meanings'], list)
        assert len(read_ds[qc_var_name].attrs['flag_meanings']) == 1
        assert isinstance(read_ds[qc_var_name].attrs['flag_assessments'], list)
        assert len(read_ds[qc_var_name].attrs['flag_assessments']) == 1
        assert read_ds[qc_var_name].attrs['flag_assessments'] == ['Bad']
        assert read_ds[qc_var_name].attrs['flag_meanings'] == ['Value is equal to missing_value.']

        read_ds.close()
        del read_ds


def test_io_mpldataset():
    try:
        mpl_ds = act.io.mpl.read_sigma_mplv5(act.tests.EXAMPLE_SIGMA_MPLV5)
    except Exception:
        return

    # Tests fields
    assert 'channel_1' in mpl_ds.variables.keys()
    assert 'temp_0' in mpl_ds.variables.keys()
    assert mpl_ds.channel_1.values.shape == (102, 1000)

    # Tests coordinates
    assert 'time' in mpl_ds.coords.keys()
    assert 'range' in mpl_ds.coords.keys()
    assert mpl_ds.coords['time'].values.shape == (102,)
    assert mpl_ds.coords['range'].values.shape == (1000,)
    assert '_arm_standards_flag' in mpl_ds.attrs.keys()

    # Tests attributes
    assert '_datastream' in mpl_ds.attrs.keys()
    mpl_ds.close()


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


def test_read_psl_wind_profiler():
    test_ds_low, test_ds_hi = act.io.noaapsl.read_psl_wind_profiler(
        act.tests.EXAMPLE_NOAA_PSL, transpose=False
    )
    # test dimensions
    assert 'time' and 'HT' in test_ds_low.dims.keys()
    assert 'time' and 'HT' in test_ds_hi.dims.keys()
    assert test_ds_low.dims['time'] == 4
    assert test_ds_hi.dims['time'] == 4
    assert test_ds_low.dims['HT'] == 49
    assert test_ds_hi.dims['HT'] == 50

    # test coordinates
    assert (
        test_ds_low.coords['HT'][0:5] == np.array([0.151, 0.254, 0.356, 0.458, 0.561])
    ).all()
    assert (
        test_ds_low.coords['time'][0:2]
        == np.array(
            ['2021-05-05T15:00:01.000000000', '2021-05-05T15:15:49.000000000'],
            dtype='datetime64[ns]',
        )
    ).all()

    # test attributes
    assert test_ds_low.attrs['site_identifier'] == 'CTD'
    assert test_ds_low.attrs['data_type'] == 'WINDS'
    assert test_ds_low.attrs['revision_number'] == '5.1'
    assert test_ds_low.attrs['latitude'] == 34.66
    assert test_ds_low.attrs['longitude'] == -87.35
    assert test_ds_low.attrs['elevation'] == 187.0
    assert (test_ds_low.attrs['beam_azimuth'] == np.array(
        [38.0, 38.0, 308.0], dtype='float32')).all()
    assert (test_ds_low.attrs['beam_elevation'] == np.array(
        [90.0, 74.7, 74.7], dtype='float32')).all()
    assert test_ds_low.attrs['consensus_average_time'] == 24
    assert test_ds_low.attrs['oblique-beam_vertical_correction'] == 0
    assert test_ds_low.attrs['number_of_beams'] == 3
    assert test_ds_low.attrs['number_of_range_gates'] == 49
    assert test_ds_low.attrs['number_of_gates_oblique'] == 49
    assert test_ds_low.attrs['number_of_gates_vertical'] == 49
    assert test_ds_low.attrs['number_spectral_averages_oblique'] == 50
    assert test_ds_low.attrs['number_spectral_averages_vertical'] == 50
    assert test_ds_low.attrs['pulse_width_oblique'] == 708
    assert test_ds_low.attrs['pulse_width_vertical'] == 708
    assert test_ds_low.attrs['inner_pulse_period_oblique'] == 50
    assert test_ds_low.attrs['inner_pulse_period_vertical'] == 50
    assert test_ds_low.attrs['full_scale_doppler_value_oblique'] == 20.9
    assert test_ds_low.attrs['full_scale_doppler_value_vertical'] == 20.9
    assert test_ds_low.attrs['delay_to_first_gate_oblique'] == 4000
    assert test_ds_low.attrs['delay_to_first_gate_vertical'] == 4000
    assert test_ds_low.attrs['spacing_of_gates_oblique'] == 708
    assert test_ds_low.attrs['spacing_of_gates_vertical'] == 708

    # test fields
    assert test_ds_low['RAD1'].shape == (4, 49)
    assert test_ds_hi['RAD1'].shape == (4, 50)
    assert (test_ds_low['RAD1'][0, 0:5] == np.array(
        [0.2, 0.1, 0.1, 0.0, -0.1])).all()
    assert (test_ds_hi['RAD1'][0, 0:5] == np.array(
        [0.1, 0.1, -0.1, 0.0, -0.2])).all()

    assert test_ds_low['SPD'].shape == (4, 49)
    assert test_ds_hi['SPD'].shape == (4, 50)
    assert (test_ds_low['SPD'][0, 0:5] == np.array(
        [2.5, 3.3, 4.3, 4.3, 4.8])).all()
    assert (test_ds_hi['SPD'][0, 0:5] == np.array(
        [3.7, 4.6, 6.3, 5.2, 6.8])).all()

    # test transpose
    test_ds_low, test_ds_hi = act.io.noaapsl.read_psl_wind_profiler(
        act.tests.EXAMPLE_NOAA_PSL, transpose=True
    )
    assert test_ds_low['RAD1'].shape == (49, 4)
    assert test_ds_hi['RAD1'].shape == (50, 4)
    assert test_ds_low['SPD'].shape == (49, 4)
    assert test_ds_hi['SPD'].shape == (50, 4)
    test_ds_low.close()


def test_read_psl_wind_profiler_temperature():
    ds = read_psl_wind_profiler_temperature(
        act.tests.EXAMPLE_NOAA_PSL_TEMPERATURE)

    ds.attrs['site_identifier'] == 'CTD'
    ds.attrs['elevation'] = 600.0
    ds.T.values[0] == 33.2


def test_read_psl_surface_met():
    ds = read_psl_surface_met(sample_files.EXAMPLE_NOAA_PSL_SURFACEMET)
    assert ds.time.size == 2
    assert np.isclose(np.sum(ds['Pressure'].values), 1446.9)
    assert np.isclose(ds['lat'].values, 38.972425)
    assert ds['lat'].attrs['units'] == 'degree_N'
    assert ds['Upward_Longwave_Irradiance'].attrs['long_name'] == 'Upward Longwave Irradiance'
    assert ds['Upward_Longwave_Irradiance'].dtype.str == '<f4'

    with pytest.raises(Exception):
        ds = read_psl_surface_met('aaa22001.00m')


def test_read_psl_parsivel():
    url = ['https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200200_stats.txt',
           'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200201_stats.txt',
           'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200202_stats.txt']

    ds = act.io.noaapsl.read_psl_parsivel(url)
    assert 'number_density_drops' in ds
    assert np.max(ds['number_density_drops'].values) == 355
    assert ds['number_density_drops'].values[10, 10] == 201

    ds = act.io.noaapsl.read_psl_parsivel(
        'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200201_stats.txt')
    assert 'number_density_drops' in ds


def test_read_psl_fmcw_moment():
    result = act.discovery.download_noaa_psl_data(
        site='kps', instrument='Radar FMCW Moment',
        startdate='20220815', hour='06'
    )
    ds = act.io.noaapsl.read_psl_radar_fmcw_moment([result[-1]])
    assert 'range' in ds
    np.testing.assert_almost_equal(
        ds['reflectivity_uncalibrated'].mean(), 2.37, decimal=2)
    assert ds['range'].max() == 10040.
    assert len(ds['time'].values) == 115


def test_read_psl_sband_moment():
    result = act.discovery.download_noaa_psl_data(
        site='ctd', instrument='Radar S-band Moment',
        startdate='20211225', hour='06'
    )
    ds = act.io.noaapsl.read_psl_radar_sband_moment([result[-1]])
    assert 'range' in ds
    np.testing.assert_almost_equal(
        ds['reflectivity_uncalibrated'].mean(), 1.00, decimal=2)
    assert ds['range'].max() == 9997.
    assert len(ds['time'].values) == 37


@pytest.mark.skipif(not act.io.icartt._ICARTT_AVAILABLE,
                    reason="ICARTT is not installed.")
def test_read_icartt():
    result = act.io.icartt.read_icartt(act.tests.EXAMPLE_AAF_ICARTT)
    assert 'pitch' in result
    assert len(result['time'].values) == 14087
    assert result['true_airspeed'].units == 'm/s'
    assert 'Revision' in result.attrs
    np.testing.assert_almost_equal(
        result['static_pressure'].mean(), 708.75, decimal=2)


def test_unpack_tar():

    with tempfile.TemporaryDirectory() as tmpdirname:

        tar_file = Path(tmpdirname, 'tar_file_dir')
        output_dir = Path(tmpdirname, 'output_dir')
        tar_file.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for tar_file_name in ['test_file1.tar', 'test_file2.tar']:
            filenames = []
            for value in range(0, 10):
                filename = "".join(random.choices(list(ascii_letters), k=15))
                filename = Path(tar_file, f"{filename}.nc")
                filename.touch()
                filenames.append(filename)
            act.utils.io_utils.pack_tar(filenames, write_filename=Path(tar_file, tar_file_name),
                                        remove=True)

        tar_files = list(tar_file.glob('*.tar'))
        result = act.utils.io_utils.unpack_tar(tar_files[0], write_directory=output_dir)
        assert isinstance(result, list)
        assert len(result) == 10
        for file in result:
            assert isinstance(file, (str, PathLike))

        files = list(output_dir.glob('*'))
        assert len(files) == 1
        assert files[0].is_dir()
        act.utils.io_utils.cleanup_files(dirname=output_dir)
        files = list(output_dir.glob('*'))
        assert len(files) == 0

        # Check not returing file but directory
        result = act.utils.io_utils.unpack_tar(tar_files[0], write_directory=output_dir, return_files=False)
        assert isinstance(result, str)
        files = list(Path(result).glob('*'))
        assert len(files) == 10
        act.utils.io_utils.cleanup_files(result)
        files = list(Path(output_dir).glob('*'))
        assert len(files) == 0

        # Test temporary directory
        result = act.utils.io_utils.unpack_tar(tar_files[0], temp_dir=True)
        assert isinstance(result, list)
        assert len(result) == 10
        for file in result:
            assert isinstance(file, (str, PathLike))

        act.utils.io_utils.cleanup_files(files=result)

        # Test removing TAR file
        result = act.utils.io_utils.unpack_tar(tar_files, write_directory=output_dir, remove=True)
        assert isinstance(result, list)
        assert len(result) == 20
        for file in result:
            assert isinstance(file, (str, PathLike))

        tar_files = list(tar_file.glob('*.tar'))
        assert len(tar_files) == 0

        act.utils.io_utils.cleanup_files(files=result)
        files = list(Path(output_dir).glob('*'))
        assert len(files) == 0

        not_a_tar_file = Path(tar_file, 'not_a_tar_file.tar')
        not_a_tar_file.touch()
        result = act.utils.io_utils.unpack_tar(not_a_tar_file, Path(output_dir, 'another_dir'))
        assert result == []

        act.utils.io_utils.cleanup_files()

        not_a_directory = '/asasfdlkjsdfjioasdflasdfhasd/not/a/directory'
        act.utils.io_utils.cleanup_files(dirname=not_a_directory)

        not_a_file = Path(not_a_directory, 'not_a_file.nc')
        act.utils.io_utils.cleanup_files(files=not_a_file)

        act.utils.io_utils.cleanup_files(files=output_dir)

        dir_names = list(Path(tmpdirname).glob('*'))
        for dir_name in [tar_file, output_dir]:
            assert dir_name, dir_name in dir_names

        filename = "".join(random.choices(list(ascii_letters), k=15))
        filename = Path(tar_file, f"{filename}.nc")
        filename.touch()
        result = act.utils.io_utils.pack_tar(
            filename, write_filename=Path(tar_file, 'test_file_single'), remove=True)
        assert Path(filename).is_file() is False
        assert Path(result).is_file()
        assert result.endswith('.tar')


def test_gunzip():

    with tempfile.TemporaryDirectory() as tmpdirname:

        filenames = []
        for value in range(0, 10):
            filename = "".join(random.choices(list(ascii_letters), k=15))
            filename = Path(tmpdirname, f"{filename}.nc")
            filename.touch()
            filenames.append(filename)

        filename = act.utils.io_utils.pack_tar(filenames, write_directory=tmpdirname, remove=True)
        files = list(Path(tmpdirname).glob('*'))
        assert len(files) == 1
        assert files[0].name == 'created_tarfile.tar'
        assert Path(filename).name == 'created_tarfile.tar'

        gzip_file = act.utils.io_utils.pack_gzip(filename=filename)
        files = list(Path(tmpdirname).glob('*'))
        assert len(files) == 2
        files = list(Path(tmpdirname).glob('*.gz'))
        assert files[0].name == 'created_tarfile.tar.gz'
        assert Path(gzip_file).name == 'created_tarfile.tar.gz'

        unpack_filename = act.utils.io_utils.unpack_gzip(filename=gzip_file)
        files = list(Path(tmpdirname).glob('*'))
        assert len(files) == 2
        assert Path(unpack_filename).name == 'created_tarfile.tar'

        result = act.utils.io_utils.unpack_tar(unpack_filename, return_files=True, randomize=True)
        files = list(Path(Path(result[0]).parent).glob('*'))
        assert len(result) == 10
        assert len(files) == 10
        for file in result:
            assert file.endswith('.nc')

    with tempfile.TemporaryDirectory() as tmpdirname:

        filenames = []
        for value in range(0, 10):
            filename = "".join(random.choices(list(ascii_letters), k=15))
            filename = Path(tmpdirname, f"{filename}.nc")
            filename.touch()
            filenames.append(filename)

        filename = act.utils.io_utils.pack_tar(filenames, write_directory=tmpdirname, remove=True)
        files = list(Path(tmpdirname).glob('*'))
        assert len(files) == 1
        files = list(Path(tmpdirname).glob('*.tar'))
        assert files[0].name == 'created_tarfile.tar'
        assert Path(filename).name == 'created_tarfile.tar'

        gzip_file = act.utils.io_utils.pack_gzip(
            filename=filename, write_directory=Path(filename).parent, remove=False)
        files = list(Path(tmpdirname).glob('*'))
        assert len(files) == 2
        files = list(Path(tmpdirname).glob('*gz'))
        assert files[0].name == 'created_tarfile.tar.gz'
        assert Path(gzip_file).name == 'created_tarfile.tar.gz'

        unpack_filename = act.utils.io_utils.unpack_gzip(
            filename=gzip_file, write_directory=Path(filename).parent, remove=False)
        files = list(Path(tmpdirname).glob('*'))
        assert len(files) == 2
        assert Path(unpack_filename).name == 'created_tarfile.tar'

        result = act.utils.io_utils.unpack_tar(unpack_filename, return_files=True, randomize=False, remove=True)
        files = list(Path(Path(result[0]).parent).glob('*.nc'))
        assert len(result) == 10
        assert len(files) == 10
        for file in result:
            assert file.endswith('.nc')

        assert Path(unpack_filename).is_file() is False


def test_read_netcdf_tarfiles():

    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = Path(act.tests.EXAMPLE_MET_WILDCARD)
        met_files = list(Path(met_files.parent).glob(met_files.name))
        filename = act.utils.io_utils.pack_tar(met_files, write_directory=tmpdirname)
        ds = act.io.armfiles.read_netcdf(filename)
        ds.clean.cleanup()

        assert 'temp_mean' in ds.data_vars


def test_read_netcdf_gztarfiles():
    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = Path(act.tests.EXAMPLE_MET_WILDCARD)
        met_files = list(Path(met_files.parent).glob(met_files.name))
        filename = act.utils.io_utils.pack_tar(met_files, write_directory=tmpdirname)
        filename = act.utils.io_utils.pack_gzip(filename, write_directory=tmpdirname, remove=True)
        ds = act.io.armfiles.read_netcdf(filename)
        ds.clean.cleanup()

        assert 'temp_mean' in ds.data_vars

    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = sample_files.EXAMPLE_MET1
        filename = act.utils.io_utils.pack_gzip(met_files, write_directory=tmpdirname, remove=False)
        ds = act.io.armfiles.read_netcdf(filename)
        ds.clean.cleanup()

        assert 'temp_mean' in ds.data_vars


def test_read_mmcr():
    results = glob.glob(act.tests.EXAMPLE_MMCR)
    ds = act.io.armfiles.read_mmcr(results)
    assert 'MeanDopplerVelocity_PR' in ds
    assert 'SpectralWidth_BL' in ds
    np.testing.assert_almost_equal(
        ds['Reflectivity_GE'].mean(), -34.62, decimal=2)
    np.testing.assert_almost_equal(
        ds['MeanDopplerVelocity_Receiver1'].max(), 9.98, decimal=2)


def test_read_neon():
    data_file = glob.glob(act.tests.EXAMPLE_NEON)
    variable_file = glob.glob(act.tests.EXAMPLE_NEON_VARIABLE)
    position_file = glob.glob(act.tests.EXAMPLE_NEON_POSITION)

    ds = act.io.neon.read_neon_csv(data_file)
    assert len(ds['time'].values) == 17280
    assert 'time' in ds
    assert 'tempSingleMean' in ds
    assert ds['tempSingleMean'].values[0] == -0.6003

    ds = act.io.neon.read_neon_csv(data_file, variable_files=variable_file, position_files=position_file)
    assert ds['northOffset'].values == -5.79
    assert ds['tempSingleMean'].attrs['units'] == 'celsius'
    assert 'lat' in ds
    assert ds['lat'].values == 71.282425


def test_read_sodar():
    ds = act.io.read_mfas_sodar(act.tests.EXAMPLE_MFAS_SODAR)

    # Test coordinates.
    assert ds.time.shape[0] == 96
    assert ds.time[0].dtype == 'datetime64[ns]'

    assert ds.height.shape[0] == 58
    assert ds.height[0] == 30.0
    assert ds.height[-1] == 600.0

    # Test variable data, shape and attributes.
    assert len(ds.data_vars) == 26
    assert ds['dir'].shape == (96, 58)
    direction = ds['dir'][0, 0:5].values
    np.testing.assert_allclose(
        direction, [129.9, 144.2, 147.5, 143.5, 143.0], rtol=1e-6)
    pgz = ds['PGz'][0, 0:5].values
    np.testing.assert_allclose(
        pgz, [4, 4, 4, 5, 5])

    assert ds['dir'].attrs['variable_name'] == 'wind direction'
    assert ds['dir'].attrs['symbol'] == 'dir'
    assert ds['dir'].attrs['type'] == 'R1'
    assert ds['dir'].attrs['_FillValue'] == 999.9
    assert ds['dir'].attrs['error_mask'] == '0'
    assert ds['dir'].attrs['units'] == 'deg'

    # Test global attributes.
    assert ds.attrs['height above sea level [m]'] == 0.0
    assert ds.attrs['instrument_type'] == 'MFAS'


def test_metadata_retrieval():
    # Read the file and lines.
    file = fsspec.open(act.tests.EXAMPLE_MFAS_SODAR).open()
    lines = file.readlines()
    lines = [x.decode().rstrip()[:] for x in lines]

    # Retrieve metadata.
    file_dict, variable_dict = act.io.sodar._metadata_retrieval(lines)

    # Test file dictionary.
    assert 'instrument_type' in file_dict
    assert 'height above sea level [m]' in file_dict

    assert file_dict['format'] == 'FORMAT-1'
    assert file_dict['height above ground [m]'] == 0.0

    # Test variable dictionary.
    assert 'speed' in variable_dict.keys()
    assert 'error' not in variable_dict.keys()

    assert 'variable_name' in variable_dict['sigSpeed']
    assert 'units' in variable_dict['sigSpeed']
    assert '_FillValue' in variable_dict['sigSpeed']

    assert variable_dict['W']['units'] == 'm/s'
    assert variable_dict['W']['variable_name'] == 'wind W'
    assert variable_dict['W']['_FillValue'] == 99.99


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
