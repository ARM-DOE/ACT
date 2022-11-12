from pathlib import Path
import tarfile
from os import PathLike
from shutil import rmtree
import random
import string
import gzip
import shutil
import tempfile


def pack_tar(filenames, write_filename=None, write_directory=None, remove=False):
    """
    Creates TAR file from list of filenames provided. Currently only works with
    all files existing in the same directory.

    ...

    Parameters
    ----------
    filenames : str or list
        Filenames to be placed in TAR file
    write_filename : str, pathlib.Path, None
        TAR output filename. If not provided will use file name 'created_tarfile.tar'
    write_directory : str, pathlib.Path, None
        Path to directory to write TAR file. If the directory does not exist will
        be created.
    remove : boolean
        Delete provided filenames after making TAR file

    Returns
    -------
    list
        List of files extracted from the TAR file or full path to created direcotry
        containing extracted files.

    """

    if write_filename is None:
        write_filename = 'created_tarfile.tar'

    if isinstance(filenames, (str, PathLike)):
        filenames = [filenames]

    if write_directory is not None:
        write_directory = Path(write_directory)
        write_directory.mkdir(parents=True, exist_ok=True)
        write_filename = Path(write_filename).name
    elif Path(write_filename).parent != Path('.'):
        write_directory = Path(write_filename).parent
    else:
        write_directory = Path('.')

    if not str(write_filename).endswith('.tar'):
        write_filename = str(write_filename) + '.tar'

    write_filename = Path(write_directory, write_filename)
    tar_file_handle = tarfile.open(write_filename, "w")
    for filename in filenames:
        tar_file_handle.add(filename, arcname=Path(filename).name)

    tar_file_handle.close()

    if remove:
        for filename in filenames:
            Path(filename).unlink()

    return str(write_filename)


def unpack_tar(tar_files, write_directory=None, temp_dir=False, randomize=True,
               return_files=True, remove=False):
    """
    Unpacks TAR file contents into provided base directory

    ...

    Parameters
    ----------
    tar_files : str or list
        path to TAR file to be unpacked
    write_directory : str or pathlib.Path
        base path to extract contents of TAR files or create a new randomized directory
        to extract contents of TAR file.
    temp_dir : boolean
        Should a temporary directory be created and TAR files extracted to the new directory.
        write_directory and randomize are ignored if this option is used.
    randomize : boolean
        Create a new randomized directory to extract TAR files into.
    return_files : boolean
        When set will return a list of full path filenames to the extracted files.
        When set to False will return full path to directory containing extracted files.
    remove : boolean
        Delete provided TAR files after extracting files.

    Returns
    -------
    files : list or str
        List of full path files extracted from the TAR file or full path to direcotry
        containing extracted files.

    """

    files = []

    if isinstance(tar_files, (str, PathLike)):
        tar_files = [tar_files]

    out_dir = Path.cwd()
    if temp_dir is True:
        out_dir = Path(tempfile.TemporaryDirectory().name)
    else:
        if write_directory is not None:
            out_dir = Path(write_directory)
        else:
            out_dir = Path(Path(tar_files[0]).parent)

        if out_dir.is_dir() is False:
            out_dir.mkdir(parents=True, exist_ok=True)

        if randomize:
            out_dir = Path(tempfile.mkdtemp(dir=out_dir))

    for tar_file in tar_files:
        try:
            tar = tarfile.open(tar_file)
            tar.extractall(path=out_dir)
            result = [str(Path(out_dir, ii.name)) for ii in tar.getmembers()]
            files.extend(result)
            tar.close()
        except tarfile.ReadError:
            print(f"\nCould not extract files from {tar_file}")

    if return_files is False:
        files = str(out_dir)
    else:
        files.sort()

    if remove:
        for tar_file in tar_files:
            Path(tar_file).unlink()

    return files


def cleanup_files(dirname=None, files=None):
    """
    Cleans up files and directory possibly created from unpacking TAR files with unpack_tar()

    ...

    Parameters
    ----------
    dirname : str, pathlib.Path, None
        Path to directory of extracted files which will be removed.
    files : str, pahtlib.Path, list, None
        Full path file name(s) from extracted TAR file.
        Assumes the directory this file exists in should be removed.

    """

    if isinstance(files, (str, PathLike)):
        files = [str(files)]

    try:
        if dirname is not None:
            rmtree(dirname)

        if files is not None and len(files) > 0 and Path(files[0]).is_file():
            out_dir = Path(files[0]).parent
            rmtree(out_dir)

    except Exception as error:
        print("\nError removing files:", error)


def is_gunzip_file(filepath):
    """
    Function to test if file is a gunzip file.

    Parameters
    ----------

    filepath : str or pathlib.Path to file to test

    Returns
    -------
    test : boolean
        Result from testing if file is a gunzip file

    """

    try:
        with open(str(filepath), 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'
    except Exception:
        return False


def pack_gzip(filename, write_directory=None, remove=False):
    """
    Creates a gunzip file from a filename path

    ...

    Parameters
    ----------
    filename : str, pathlib.Path
        Filename to use in creation of gunzip version.
    write_directory : str, pahtlib.Path, list, None
        Path to directory to place newly created gunzip file.
    remove : boolean
        Remove provided filename after creating gunzip file

    Returns
    -------
    write_filename : str
        Full path name of created gunzip file

    """

    write_filename = Path(filename).name + '.gz'

    if write_directory is not None:
        write_filename = Path(write_directory, write_filename)
        Path(write_directory).mkdir(parents=True, exist_ok=True)
    else:
        write_filename = Path(Path(filename).parent, write_filename)

    with open(filename, 'rb') as f_in:
        with gzip.open(write_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if remove:
        Path(filename).unlink()

    return str(write_filename)


def unpack_gzip(filename, write_directory=None, remove=False):
    """
    Extracts file from a gunzip file.

    ...

    Parameters
    ----------
    filename : str, pathlib.Path
        Filename to use in extraction of gunzip file.
    write_directory : str, pahtlib.Path, list, None
        Path to directory to place newly created gunzip file.
    remove : boolean
        Remove provided filename after creating gunzip file

    Returns
    -------
    write_filename : str
        Full path name of created gunzip file

    """

    if write_directory is None:
        write_directory = Path(filename).parent

    write_filename = Path(filename).name
    if write_filename.endswith('.gz'):
        write_filename = write_filename.replace(".gz", "")

    write_filename = Path(write_directory, write_filename)

    with gzip.open(filename, "rb") as f_in:
        with open(write_filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if remove:
        Path(filename).unlink()

    return str(write_filename)
