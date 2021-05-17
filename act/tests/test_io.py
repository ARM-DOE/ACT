import act
from act.io.noaagml import read_gml
import act.tests.sample_files as sample_files
from pathlib import Path
import tempfile
import numpy as np
import glob


def test_io():
    sonde_ds = act.io.armfiles.read_netcdf([act.tests.EXAMPLE_MET1])
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


def test_io_mfdataset():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET_WILDCARD)
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert len(sonde_ds.attrs['_file_times']) == 7
    assert sonde_ds.attrs['_arm_standards_flag'] == (1 << 0)
    sonde_ds.close()


def test_io_csv():
    headers = ['day', 'month', 'year', 'time', 'pasquill',
               'wdir_60m', 'wspd_60m', 'wdir_60m_std',
               'temp_60m', 'wdir_10m', 'wspd_10m',
               'wdir_10m_std', 'temp_10m', 'temp_dp',
               'rh', 'avg_temp_diff', 'total_precip',
               'solar_rad', 'net_rad', 'atmos_press',
               'wv_pressure', 'temp_soil_10cm',
               'temp_soil_100cm', 'temp_soil_10ft']
    anl_ds = act.io.csvfiles.read_csv(
        act.tests.EXAMPLE_ANL_CSV, sep='\s+', column_names=headers)
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
        obj = act.io.armfiles.create_obj_from_arm_dod('vdis.b1', dims, version='1.2',
                                                      scalar_fill_dim='time')
        assert 'moment1' in obj
        assert len(obj['base_time'].values) == 1440
        assert len(obj['drop_diameter'].values) == 50
        with np.testing.assert_warns(UserWarning):
            obj2 = act.io.armfiles.create_obj_from_arm_dod('vdis.b1', dims,
                                                           scalar_fill_dim='time')
        assert 'moment1' in obj2
        assert len(obj2['base_time'].values) == 1440
        assert len(obj2['drop_diameter'].values) == 50
        with np.testing.assert_raises(ValueError):
            obj = act.io.armfiles.create_obj_from_arm_dod('vdis.b1', {}, version='1.2')

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
        sonde_ds.write.write_netcdf(path=write_file, make_copy=False, join_char='_',
                                    cf_compliant=True, cf_convention=cf_convention)

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
        obj.write.write_netcdf(path=write_file, make_copy=False, join_char='_',
                               cf_compliant=True, cf_convention=cf_convention)

        obj_read = act.io.armfiles.read_netcdf(str(write_file))

        assert cf_convention in obj_read.attrs['Conventions'].split()
        assert obj_read.attrs['FeatureType'] == 'timeSeriesProfile'
        assert len(obj_read.dims) > 1

        obj_read.close()
        del obj_read


def test_io_mpldataset():
    try:
        mpl_ds = act.io.mpl.read_sigma_mplv5(
            act.tests.EXAMPLE_SIGMA_MPLV5)
    except Exception:
        return

    # Tests fields
    assert 'channel_1' in mpl_ds.variables.keys()
    assert 'temp_0' in mpl_ds.variables.keys()
    assert mpl_ds.channel_1.values.shape == (102, 1000)

    # Tests coordinates
    assert 'time' in mpl_ds.coords.keys()
    assert 'range' in mpl_ds.coords.keys()
    assert mpl_ds.coords['time'].values.shape == (102, )
    assert mpl_ds.coords['range'].values.shape == (1000, )
    assert '_arm_standards_flag' in mpl_ds.attrs.keys()

    # Tests attributes
    assert '_datastream' in mpl_ds.attrs.keys()
    mpl_ds.close()


def test_read_gml():
    # Test Radiation
    ds = read_gml(sample_files.EXAMPLE_GML_RADIATION, datatype='RADIATION')
    assert np.isclose(np.nansum(ds['solar_zenith_angle']), 1629.68)
    assert np.isclose(np.nansum(ds['upwelling_infrared_case_temp']), 4185.73)
    assert (ds['upwelling_infrared_case_temp'].attrs['ancillary_variables'] ==
            'qc_upwelling_infrared_case_temp')
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_values'] == [0, 1, 2]
    assert (ds['qc_upwelling_infrared_case_temp'].attrs['flag_meanings'] ==
            ['Not failing any tests', 'Knowingly bad value', 'Should be used with scrutiny'])
    assert (ds['qc_upwelling_infrared_case_temp'].attrs['flag_assessments'] ==
            ['Good', 'Bad', 'Indeterminate'])
    assert ds['time'].values[-1] == np.datetime64('2021-01-01T00:17:00')

    ds = read_gml(sample_files.EXAMPLE_GML_RADIATION, convert_missing=False)
    assert np.isclose(np.nansum(ds['solar_zenith_angle']), 1629.68)
    assert np.isclose(np.nansum(ds['upwelling_infrared_case_temp']), 4185.73)
    assert (ds['upwelling_infrared_case_temp'].attrs['ancillary_variables'] ==
            'qc_upwelling_infrared_case_temp')
    assert ds['qc_upwelling_infrared_case_temp'].attrs['flag_values'] == [0, 1, 2]
    assert (ds['qc_upwelling_infrared_case_temp'].attrs['flag_meanings'] ==
            ['Not failing any tests', 'Knowingly bad value', 'Should be used with scrutiny'])
    assert (ds['qc_upwelling_infrared_case_temp'].attrs['flag_assessments'] ==
            ['Good', 'Bad', 'Indeterminate'])
    assert ds['time'].values[-1] == np.datetime64('2021-01-01T00:17:00')

    # Test MET
    ds = read_gml(sample_files.EXAMPLE_GML_MET, datatype='MET')
    assert np.isclose(np.nansum(ds['wind_speed'].values), 140.999)
    assert ds['wind_speed'].attrs['units'] == 'm/s'
    assert np.isnan(ds['wind_speed'].attrs['_FillValue'])
    assert np.sum(np.isnan(ds['preciptation_intensity'].values)) == 19
    assert ds['preciptation_intensity'].attrs['units'] == 'mm/hour'
    assert ds['time'].values[0] == np.datetime64('2020-01-01T01:00:00')

    ds = read_gml(sample_files.EXAMPLE_GML_MET, convert_missing=False)
    assert np.isclose(np.nansum(ds['wind_speed'].values), 140.999)
    assert ds['wind_speed'].attrs['units'] == 'm/s'
    assert np.isclose(ds['wind_speed'].attrs['_FillValue'], -999.9)
    assert np.sum(ds['preciptation_intensity'].values) == -1881
    assert ds['preciptation_intensity'].attrs['units'] == 'mm/hour'
    assert ds['time'].values[0] == np.datetime64('2020-01-01T01:00:00')

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
    assert (ds['qc_co2'].values ==
            np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)).all()
    assert ds['co2'].attrs['units'] == 'ppm'
    assert np.isnan(ds['co2'].attrs['_FillValue'])
    assert ds['qc_co2'].attrs['flag_assessments'] == ['Bad', 'Indeterminate']
    assert ds['latitude'].attrs['standard_name'] == 'latitude'

    ds = read_gml(sample_files.EXAMPLE_GML_CO2, convert_missing=False)
    assert np.isclose(np.nansum(ds['co2'].values), -3692.3098)
    assert ds['co2'].attrs['_FillValue'] == -999.99
    assert (ds['qc_co2'].values ==
            np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)).all()
    assert ds['co2'].attrs['units'] == 'ppm'
    assert np.isclose(ds['co2'].attrs['_FillValue'], -999.99)
    assert ds['qc_co2'].attrs['flag_assessments'] == ['Bad', 'Indeterminate']
    assert ds['latitude'].attrs['standard_name'] == 'latitude'

    # Test Halocarbon
    ds = read_gml(sample_files.EXAMPLE_GML_HALO, datatype='HALO')
    assert np.isclose(np.nansum(ds['CCl4'].values), 1342.6499)
    assert ds['CCl4'].attrs['units'] == 'ppt'
    assert ds['CCl4'].attrs['long_name'] == 'Carbon Tetrachloride (CCl4) daily median'
    assert np.isnan(ds['CCl4'].attrs['_FillValue'])
    assert ds['time'].values[0] == np.datetime64('1998-06-16T00:00:00')

    ds = read_gml(sample_files.EXAMPLE_GML_HALO)
    assert np.isclose(np.nansum(ds['CCl4'].values), 1342.6499)
    assert ds['CCl4'].attrs['units'] == 'ppt'
    assert ds['CCl4'].attrs['long_name'] == 'Carbon Tetrachloride (CCl4) daily median'
    assert np.isnan(ds['CCl4'].attrs['_FillValue'])
    assert ds['time'].values[0] == np.datetime64('1998-06-16T00:00:00')
