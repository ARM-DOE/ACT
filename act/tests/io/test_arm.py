from os import PathLike
from pathlib import Path
import random
from string import ascii_letters
import tempfile

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
    assert ds['time'].values[10].astype('datetime64[ms]') == np.datetime64('2019-01-01T00:10:00', 'ms')

    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET1, use_base_time=True, drop_variables='time')
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
    drop_variables = act.io.arm.keep_variables_to_drop_variables(
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
    np.testing.assert_almost_equal(
        ds['Reflectivity_GE'].mean(), -34.62, decimal=2)
    np.testing.assert_almost_equal(
        ds['MeanDopplerVelocity_Receiver1'].max(), 9.98, decimal=2)


def test_read_netcdf_gztarfiles():
    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = list(Path(file) for file in act.tests.EXAMPLE_MET_WILDCARD)
        filename = act.utils.io_utils.pack_tar(met_files, write_directory=tmpdirname)
        filename = act.utils.io_utils.pack_gzip(filename, write_directory=tmpdirname, remove=True)
        ds = act.io.arm.read_arm_netcdf(filename)
        ds.clean.cleanup()

        assert 'temp_mean' in ds.data_vars

    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = sample_files.EXAMPLE_MET1
        filename = act.utils.io_utils.pack_gzip(met_files, write_directory=tmpdirname, remove=False)
        ds = act.io.arm.read_arm_netcdf(filename)
        ds.clean.cleanup()

        assert 'temp_mean' in ds.data_vars


def test_read_netcdf_tarfiles():
    with tempfile.TemporaryDirectory() as tmpdirname:
        met_files = list(Path(file) for file in act.tests.EXAMPLE_MET_WILDCARD)
        filename = act.utils.io_utils.pack_tar(met_files, write_directory=tmpdirname)
        ds = act.io.arm.read_arm_netcdf(filename)
        ds.clean.cleanup()

        assert 'temp_mean' in ds.data_vars


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
