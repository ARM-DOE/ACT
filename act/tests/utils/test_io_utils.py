import tempfile
from pathlib import Path
from os import PathLike
from string import ascii_letters
import random

import act
from act.tests import sample_files

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
