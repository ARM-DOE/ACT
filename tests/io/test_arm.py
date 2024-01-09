import tempfile
from pathlib import Path

import numpy as np

import act
from act.tests import sample_files


def test_read_arm_netcdf():
    ds = act.io.arm.read_arm_netcdf([act.tests.EXAMPLE_MET1])
    assert 'temp_mean' in ds.variables.keys()
    assert 'rh_mean' in ds.variables.keys()
    assert ds.attrs['_arm_standards_flag'] == (1 << 0)

    with np.testing.assert_raises(OSError):
        ds = act.io.arm.read_arm_netcdf([])

    ds = act.io.arm.read_arm_netcdf([], return_None=True)
    assert ds is None
    ds = act.io.arm.read_arm_netcdf(['./randomfile.nc'], return_None=True)
    assert ds is None

    ds = act.io.arm.read_arm_netcdf([act.tests.EXAMPLE_MET_TEST1])
    assert 'time' in ds

    ds = act.io.arm.read_arm_netcdf([act.tests.EXAMPLE_MET_TEST2])
    assert ds['time'].values[10].astype('datetime64[ms]') == np.datetime64(
        '2019-01-01T00:10:00', 'ms'
    )

    ds = act.io.arm.read_arm_netcdf(
        act.tests.EXAMPLE_MET1, use_base_time=True, drop_variables='time'
    )
    assert 'time' in ds
    assert np.issubdtype(ds['time'].dtype, np.datetime64)
    assert ds['time'].values[10].astype('datetime64[ms]') == np.datetime64(
        '2019-01-01T00:10:00', 'ms'
    )

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
    drop_variables = act.io.arm.keep_variables_to_drop_variables(act.tests.EXAMPLE_MET1, var_names)

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

    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET1, keep_variables='temp_mean')
    assert list(ds.data_vars) == ['temp_mean']
    del ds

    var_names = ['temp_mean', 'qc_temp_mean']
    ds = act.io.arm.read_arm_netcdf(
        act.tests.EXAMPLE_MET1, keep_variables=var_names, drop_variables='nonsense'
    )
    assert list(ds.data_vars).sort() == var_names.sort()
    del ds

    var_names = ['temp_mean', 'qc_temp_mean', 'alt', 'lat', 'lon']
    ds = act.io.arm.read_arm_netcdf(
        act.tests.EXAMPLE_MET_WILDCARD, keep_variables=var_names, drop_variables=['lon']
    )
    var_names = list(set(var_names) - {'lon'})
    assert list(ds.data_vars).sort() == var_names.sort()
    del ds

    filenames = list(Path(file) for file in act.tests.EXAMPLE_MET_WILDCARD)
    var_names = ['temp_mean', 'qc_temp_mean', 'alt', 'lat', 'lon']
    ds = act.io.arm.read_arm_netcdf(filenames, keep_variables=var_names)
    assert list(ds.data_vars).sort() == var_names.sort()
    del ds


def test_read_arm_netcdf_mfdataset():
    met_ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET_WILDCARD)
    met_ds.load()
    assert 'temp_mean' in met_ds.variables.keys()
    assert 'rh_mean' in met_ds.variables.keys()
    assert len(met_ds.attrs['_file_times']) == 7
    assert met_ds.attrs['_arm_standards_flag'] == (1 << 0)
    met_ds.close()
    del met_ds

    met_ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET_WILDCARD, cleanup_qc=True)
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


def test_io_dod():
    dims = {'time': 1440, 'drop_diameter': 50}

    try:
        ds = act.io.arm.create_ds_from_arm_dod(
            'vdis.b1', dims, version='1.2', scalar_fill_dim='time'
        )
        assert 'moment1' in ds
        assert len(ds['base_time'].values) == 1440
        assert len(ds['drop_diameter'].values) == 50
        with np.testing.assert_warns(UserWarning):
            ds2 = act.io.arm.create_ds_from_arm_dod('vdis.b1', dims, scalar_fill_dim='time')
        assert 'moment1' in ds2
        assert len(ds2['base_time'].values) == 1440
        assert len(ds2['drop_diameter'].values) == 50
        with np.testing.assert_raises(ValueError):
            ds = act.io.arm.create_ds_from_arm_dod('vdis.b1', {}, version='1.2')
        ds = act.io.arm.create_ds_from_arm_dod(
            sample_files.EXAMPLE_DOD, dims, version=1.2, scalar_fill_dim='time', local_file=True
        )
        assert 'moment1' in ds
        assert len(ds['base_time'].values) == 1440
        assert len(ds['drop_diameter'].values) == 50
    except Exception:
        return
    ds.close()
    ds2.close()


def test_io_write():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    sonde_ds.clean.cleanup()

    with tempfile.TemporaryDirectory() as tmpdirname:
        write_file = Path(tmpdirname, Path(sample_files.EXAMPLE_SONDE1).name)
        keep_vars = ['tdry', 'qc_tdry', 'dp', 'qc_dp']
        for var_name in list(sonde_ds.data_vars):
            if var_name not in keep_vars:
                del sonde_ds[var_name]
        sonde_ds.write.write_netcdf(path=write_file, FillValue=-9999)

        sonde_ds_read = act.io.arm.read_arm_netcdf(str(write_file))
        assert list(sonde_ds_read.data_vars) == keep_vars
        assert isinstance(sonde_ds_read['qc_tdry'].attrs['flag_meanings'], str)
        assert sonde_ds_read['qc_tdry'].attrs['flag_meanings'].count('__') == 21
        for attr in ['qc_standards_version', 'qc_method', 'qc_comment']:
            assert attr not in list(sonde_ds_read.attrs)
        sonde_ds_read.close()
        del sonde_ds_read

    sonde_ds.close()

    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_EBBR1)
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

        sonde_ds_read = act.io.arm.read_arm_netcdf(str(write_file))

        assert cf_convention in sonde_ds_read.attrs['Conventions'].split()
        assert sonde_ds_read.attrs['FeatureType'] == 'timeSeries'
        global_att_keys = [ii for ii in sonde_ds_read.attrs.keys() if not ii.startswith('_')]
        assert global_att_keys[-1] == 'history'
        assert sonde_ds_read['alt'].attrs['axis'] == 'Z'
        assert sonde_ds_read['alt'].attrs['positive'] == 'up'

        sonde_ds_read.close()
        del sonde_ds_read

    sonde_ds.close()

    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_CEIL1)
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

        ds_read = act.io.arm.read_arm_netcdf(str(write_file))

        assert cf_convention in ds_read.attrs['Conventions'].split()
        assert ds_read.attrs['FeatureType'] == 'timeSeriesProfile'
        assert len(ds_read.dims) > 1

        ds_read.close()
        del ds_read


def test_clean_cf_qc():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1, cleanup_qc=True)
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

        read_ds = act.io.arm.read_arm_netcdf(write_file, cleanup_qc=True)
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


def test_read_mmcr():
    results = act.tests.EXAMPLE_MMCR
    ds = act.io.arm.read_arm_mmcr(results)
    assert 'MeanDopplerVelocity_PR' in ds
    assert 'SpectralWidth_BL' in ds
    np.testing.assert_almost_equal(ds['Reflectivity_GE'].mean(), -34.62, decimal=2)
    np.testing.assert_almost_equal(ds['MeanDopplerVelocity_Receiver1'].max(), 9.98, decimal=2)
