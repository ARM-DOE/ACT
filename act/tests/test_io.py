import glob
import tempfile
from pathlib import Path
import random
import numpy as np
import pytest
from string import ascii_letters
from os import PathLike
import act
import act.tests.sample_files as sample_files
from act.io import read_gml, read_psl_wind_profiler_temperature, icartt
from act.io.noaapsl import read_psl_surface_met


def test_io():
    sonde_ds = act.io.armfiles.read_netcdf([act.tests.EXAMPLE_MET1])
    assert 'qcfilter' in list(dir(sonde_ds))
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert sonde_ds.attrs['_arm_standards_flag'] == (1 << 0)

    with np.testing.assert_raises(OSError):
        act.io.armfiles.read_netcdf([])

    result = act.io.armfiles.read_netcdf([], return_None=True)
    assert result is None
    result = act.io.armfiles.read_netcdf(['./randomfile.nc'], return_None=True)
    assert result is None

    obj = act.io.armfiles.read_netcdf([act.tests.EXAMPLE_MET_TEST1])
    assert 'time' in obj

    obj = act.io.armfiles.read_netcdf([act.tests.EXAMPLE_MET_TEST2])
    assert obj['time'].values[10] == np.datetime64('2019-01-01T00:10:00')
    sonde_ds.close()


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

    ds_object = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET1, keep_variables='temp_mean')
    assert list(ds_object.data_vars) == ['temp_mean']
    del ds_object

    var_names = ['temp_mean', 'qc_temp_mean']
    ds_object = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET1, keep_variables=var_names, drop_variables='nonsense'
    )
    assert list(ds_object.data_vars).sort() == var_names.sort()
    del ds_object

    var_names = ['temp_mean', 'qc_temp_mean', 'alt', 'lat', 'lon']
    ds_object = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET_WILDCARD, keep_variables=var_names, drop_variables=['lon']
    )
    var_names = list(set(var_names) - {'lon'})
    assert list(ds_object.data_vars).sort() == var_names.sort()
    del ds_object

    filenames = Path(act.tests.EXAMPLE_MET_WILDCARD).parent
    filenames = list(filenames.glob(Path(act.tests.EXAMPLE_MET_WILDCARD).name))
    var_names = ['temp_mean', 'qc_temp_mean', 'alt', 'lat', 'lon']
    ds_object = act.io.armfiles.read_netcdf(filenames, keep_variables=var_names)
    assert list(ds_object.data_vars).sort() == var_names.sort()
    del ds_object


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
    obj = act.io.csvfiles.read_csv(files[0])
    assert 'date_time' in obj
    assert '_datastream' in obj.attrs


def test_io_dod():
    dims = {'time': 1440, 'drop_diameter': 50}

    try:
        obj = act.io.armfiles.create_obj_from_arm_dod(
            'vdis.b1', dims, version='1.2', scalar_fill_dim='time'
        )
        assert 'moment1' in obj
        assert len(obj['base_time'].values) == 1440
        assert len(obj['drop_diameter'].values) == 50
        with np.testing.assert_warns(UserWarning):
            obj2 = act.io.armfiles.create_obj_from_arm_dod('vdis.b1', dims, scalar_fill_dim='time')
        assert 'moment1' in obj2
        assert len(obj2['base_time'].values) == 1440
        assert len(obj2['drop_diameter'].values) == 50
        with np.testing.assert_raises(ValueError):
            obj = act.io.armfiles.create_obj_from_arm_dod('vdis.b1', {}, version='1.2')
        obj = act.io.armfiles.create_obj_from_arm_dod(
            sample_files.EXAMPLE_DOD, dims, version=1.2, scalar_fill_dim='time',
            local_file=True)
        assert 'moment1' in obj
        assert len(obj['base_time'].values) == 1440
        assert len(obj['drop_diameter'].values) == 50
    except Exception:
        return
    obj.close()
    obj2.close()


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

    obj = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_CEIL1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        cf_convention = 'CF-1.8'
        write_file = Path(tmpdirname, Path(sample_files.EXAMPLE_CEIL1).name)
        obj.write.write_netcdf(
            path=write_file,
            make_copy=False,
            join_char='_',
            cf_compliant=True,
            cf_convention=cf_convention,
        )

        obj_read = act.io.armfiles.read_netcdf(str(write_file))

        assert cf_convention in obj_read.attrs['Conventions'].split()
        assert obj_read.attrs['FeatureType'] == 'timeSeriesProfile'
        assert len(obj_read.dims) > 1

        obj_read.close()
        del obj_read


def test_clean_cf_qc():
    with tempfile.TemporaryDirectory() as tmpdirname:
        obj = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_MET1, cleanup_qc=True)
        obj.load()
        var_name = 'temp_mean'
        qc_var_name = 'qc_' + var_name
        obj.qcfilter.remove_test(var_name, test_number=4)
        obj.qcfilter.remove_test(var_name, test_number=3)
        obj.qcfilter.remove_test(var_name, test_number=2)
        obj[qc_var_name].attrs['flag_masks'] = obj[qc_var_name].attrs['flag_masks'][0]
        flag_meanings = obj[qc_var_name].attrs['flag_meanings'][0]
        obj[qc_var_name].attrs['flag_meanings'] = flag_meanings.replace(' ', '__')
        flag_meanings = obj[qc_var_name].attrs['flag_assessments'][0]
        obj[qc_var_name].attrs['flag_assessments'] = flag_meanings.replace(' ', '__')

        write_file = str(Path(tmpdirname, Path(sample_files.EXAMPLE_MET1).name))
        obj.write.write_netcdf(path=write_file, cf_compliant=True)
        obj.close()
        del obj

        read_obj = act.io.armfiles.read_netcdf(write_file, cleanup_qc=True)
        read_obj.load()

        assert type(read_obj[qc_var_name].attrs['flag_masks']).__module__ == 'numpy'
        assert read_obj[qc_var_name].attrs['flag_masks'].size == 1
        assert read_obj[qc_var_name].attrs['flag_masks'][0] == 1
        assert isinstance(read_obj[qc_var_name].attrs['flag_meanings'], list)
        assert len(read_obj[qc_var_name].attrs['flag_meanings']) == 1
        assert isinstance(read_obj[qc_var_name].attrs['flag_assessments'], list)
        assert len(read_obj[qc_var_name].attrs['flag_assessments']) == 1
        assert read_obj[qc_var_name].attrs['flag_assessments'] == ['Bad']
        assert read_obj[qc_var_name].attrs['flag_meanings'] == ['Value is equal to missing_value.']

        read_obj.close()
        del read_obj


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
    test_obj_low, test_obj_hi = act.io.noaapsl.read_psl_wind_profiler(
        act.tests.EXAMPLE_NOAA_PSL, transpose=False
    )
    # test dimensions
    assert 'time' and 'HT' in test_obj_low.dims.keys()
    assert 'time' and 'HT' in test_obj_hi.dims.keys()
    assert test_obj_low.dims['time'] == 4
    assert test_obj_hi.dims['time'] == 4
    assert test_obj_low.dims['HT'] == 49
    assert test_obj_hi.dims['HT'] == 50

    # test coordinates
    assert (
        test_obj_low.coords['HT'][0:5] == np.array([0.151, 0.254, 0.356, 0.458, 0.561])
    ).all()
    assert (
        test_obj_low.coords['time'][0:2]
        == np.array(
            ['2021-05-05T15:00:01.000000000', '2021-05-05T15:15:49.000000000'],
            dtype='datetime64[ns]',
        )
    ).all()

    # test attributes
    assert test_obj_low.attrs['site_identifier'] == 'CTD'
    assert test_obj_low.attrs['data_type'] == 'WINDS'
    assert test_obj_low.attrs['revision_number'] == '5.1'
    assert test_obj_low.attrs['latitude'] == 34.66
    assert test_obj_low.attrs['longitude'] == -87.35
    assert test_obj_low.attrs['elevation'] == 187.0
    assert (test_obj_low.attrs['beam_azimuth'] == np.array(
        [38.0, 38.0, 308.0], dtype='float32')).all()
    assert (test_obj_low.attrs['beam_elevation'] == np.array(
        [90.0, 74.7, 74.7], dtype='float32')).all()
    assert test_obj_low.attrs['consensus_average_time'] == 24
    assert test_obj_low.attrs['oblique-beam_vertical_correction'] == 0
    assert test_obj_low.attrs['number_of_beams'] == 3
    assert test_obj_low.attrs['number_of_range_gates'] == 49
    assert test_obj_low.attrs['number_of_gates_oblique'] == 49
    assert test_obj_low.attrs['number_of_gates_vertical'] == 49
    assert test_obj_low.attrs['number_spectral_averages_oblique'] == 50
    assert test_obj_low.attrs['number_spectral_averages_vertical'] == 50
    assert test_obj_low.attrs['pulse_width_oblique'] == 708
    assert test_obj_low.attrs['pulse_width_vertical'] == 708
    assert test_obj_low.attrs['inner_pulse_period_oblique'] == 50
    assert test_obj_low.attrs['inner_pulse_period_vertical'] == 50
    assert test_obj_low.attrs['full_scale_doppler_value_oblique'] == 20.9
    assert test_obj_low.attrs['full_scale_doppler_value_vertical'] == 20.9
    assert test_obj_low.attrs['delay_to_first_gate_oblique'] == 4000
    assert test_obj_low.attrs['delay_to_first_gate_vertical'] == 4000
    assert test_obj_low.attrs['spacing_of_gates_oblique'] == 708
    assert test_obj_low.attrs['spacing_of_gates_vertical'] == 708

    # test fields
    assert test_obj_low['RAD1'].shape == (4, 49)
    assert test_obj_hi['RAD1'].shape == (4, 50)
    assert (test_obj_low['RAD1'][0, 0:5] == np.array(
        [0.2, 0.1, 0.1, 0.0, -0.1])).all()
    assert (test_obj_hi['RAD1'][0, 0:5] == np.array(
        [0.1, 0.1, -0.1, 0.0, -0.2])).all()

    assert test_obj_low['SPD'].shape == (4, 49)
    assert test_obj_hi['SPD'].shape == (4, 50)
    assert (test_obj_low['SPD'][0, 0:5] == np.array(
        [2.5, 3.3, 4.3, 4.3, 4.8])).all()
    assert (test_obj_hi['SPD'][0, 0:5] == np.array(
        [3.7, 4.6, 6.3, 5.2, 6.8])).all()

    # test transpose
    test_obj_low, test_obj_hi = act.io.noaapsl.read_psl_wind_profiler(
        act.tests.EXAMPLE_NOAA_PSL, transpose=True
    )
    assert test_obj_low['RAD1'].shape == (49, 4)
    assert test_obj_hi['RAD1'].shape == (50, 4)
    assert test_obj_low['SPD'].shape == (49, 4)
    assert test_obj_hi['SPD'].shape == (50, 4)
    test_obj_low.close()


def test_read_psl_wind_profiler_temperature():
    ds = read_psl_wind_profiler_temperature(
        act.tests.EXAMPLE_NOAA_PSL_TEMPERATURE)

    ds.attrs['site_identifier'] == 'CTD'
    ds.attrs['elevation'] = 600.0
    ds.T.values[0] == 33.2


def test_read_psl_surface_met():
    ds_object = read_psl_surface_met(sample_files.EXAMPLE_NOAA_PSL_SURFACEMET)
    assert ds_object.time.size == 2
    assert np.isclose(np.sum(ds_object['Pressure'].values), 1446.9)
    assert np.isclose(ds_object['lat'].values, 38.972425)
    assert ds_object['lat'].attrs['units'] == 'degree_N'
    assert ds_object['Upward_Longwave_Irradiance'].attrs['long_name'] == 'Upward Longwave Irradiance'
    assert ds_object['Upward_Longwave_Irradiance'].dtype.str == '<f4'

    with pytest.raises(Exception):
        ds_object = read_psl_surface_met('aaa22001.00m')


def test_read_psl_parsivel():
    url = ['https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200200_stats.txt',
           'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200201_stats.txt',
           'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200202_stats.txt']

    obj = act.io.noaapsl.read_psl_parsivel(url)
    assert 'number_density_drops' in obj
    assert np.max(obj['number_density_drops'].values) == 355
    assert obj['number_density_drops'].values[10, 10] == 201

    obj = act.io.noaapsl.read_psl_parsivel(
        'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200201_stats.txt')
    assert 'number_density_drops' in obj


def test_read_psl_fmcw_moment():
    result = act.discovery.download_noaa_psl_data(
        site='kps', instrument='Radar FMCW Moment',
        startdate='20220815', hour='06'
    )
    obj = act.io.noaapsl.read_psl_radar_fmcw_moment([result[-1]])
    assert 'range' in obj
    np.testing.assert_almost_equal(
        obj['reflectivity_uncalibrated'].mean(), 2.37, decimal=2)
    assert obj['range'].max() == 10040.
    assert len(obj['time'].values) == 115


def test_read_psl_sband_moment():
    result = act.discovery.download_noaa_psl_data(
        site='ctd', instrument='Radar S-band Moment',
        startdate='20211225', hour='06'
    )
    obj = act.io.noaapsl.read_psl_radar_sband_moment([result[-1]])
    assert 'range' in obj
    np.testing.assert_almost_equal(
        obj['reflectivity_uncalibrated'].mean(), 1.00, decimal=2)
    assert obj['range'].max() == 9997.
    assert len(obj['time'].values) == 37


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
        assert files[1].name == 'created_tarfile.tar.gz'
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
        assert files[0].name == 'created_tarfile.tar'
        assert Path(filename).name == 'created_tarfile.tar'

        gzip_file = act.utils.io_utils.pack_gzip(
            filename=filename, write_directory=Path(filename).parent, remove=False)
        files = list(Path(tmpdirname).glob('*'))
        assert len(files) == 2
        assert files[1].name == 'created_tarfile.tar.gz'
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
        ds_object = act.io.armfiles.read_netcdf(filename)
        ds_object.clean.cleanup()

        assert 'temp_mean' in ds_object.data_vars


def test_read_netcdf_gztarfiles():
    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = Path(act.tests.EXAMPLE_MET_WILDCARD)
        met_files = list(Path(met_files.parent).glob(met_files.name))
        filename = act.utils.io_utils.pack_tar(met_files, write_directory=tmpdirname)
        filename = act.utils.io_utils.pack_gzip(filename, write_directory=tmpdirname, remove=True)
        ds_object = act.io.armfiles.read_netcdf(filename)
        ds_object.clean.cleanup()

        assert 'temp_mean' in ds_object.data_vars

    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = sample_files.EXAMPLE_MET1
        filename = act.utils.io_utils.pack_gzip(met_files, write_directory=tmpdirname, remove=False)
        ds_object = act.io.armfiles.read_netcdf(filename)
        ds_object.clean.cleanup()

        assert 'temp_mean' in ds_object.data_vars


def test_read_mmcr():
    results = glob.glob(act.tests.EXAMPLE_MMCR)
    obj = act.io.armfiles.read_mmcr(results)
    assert 'MeanDopplerVelocity_PR' in obj
    assert 'SpectralWidth_BL' in obj
    np.testing.assert_almost_equal(
        obj['Reflectivity_GE'].mean(), -34.62, decimal=2)
    np.testing.assert_almost_equal(
        obj['MeanDopplerVelocity_Receiver1'].max(), 9.98, decimal=2)


def test_read_neon():
    data_file = glob.glob(act.tests.EXAMPLE_NEON)
    variable_file = glob.glob(act.tests.EXAMPLE_NEON_VARIABLE)
    position_file = glob.glob(act.tests.EXAMPLE_NEON_POSITION)

    obj = act.io.neon.read_neon_csv(data_file)
    assert len(obj['time'].values) == 17280
    assert 'time' in obj
    assert 'tempSingleMean' in obj
    assert obj['tempSingleMean'].values[0] == -0.6003

    obj = act.io.neon.read_neon_csv(data_file, variable_files=variable_file, position_files=position_file)
    assert obj['northOffset'].values == -5.79
    assert obj['tempSingleMean'].attrs['units'] == 'celsius'
    assert 'lat' in obj
    assert obj['lat'].values == 71.282425
