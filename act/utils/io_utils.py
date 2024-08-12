import act
import numpy as np
import pandas as pd
from pathlib import Path
import tarfile
from os import sep
from os import PathLike
from shutil import rmtree
import gzip
import shutil
import tempfile
import types

try:
    import moviepy.editor as moviepy_editor
    import moviepy.video.io.ImageSequenceClip

    MOVIEPY_AVAILABLE = True
except (ImportError, RuntimeError):
    MOVIEPY_AVAILABLE = False


def pack_tar(filenames, write_filename=None, write_directory=None, remove=False):
    """
    Creates TAR file from list of filenames provided. Currently only works with
    all files existing in the same directory.

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


def unpack_tar(
    tar_files, write_directory=None, temp_dir=False, randomize=True, return_files=True, remove=False
):
    """
    Unpacks TAR file contents into provided base directory

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
            print("Could not extract files from the tar_file")

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


def generate_movie(images, write_filename=None, fps=10, **kwargs):
    """
    Creates a movie from a list of images or convert movie to different type

    Parameters
    ----------
    images : list, PosixPath generator, path to a directory, single string/PosixPath to movie
        List of images in the correct order to make into a movie or a generator from
        a pathlib.Path.glob() search. If a path to directory will create movie from all files
        in that directory in alpanumeric order. If a single file to a movie will allow for converting
        to new format defined by file extension of write_filename.
    write_filename : str, pathlib.Path, None
        Movie output filename. Default is 'movie.mp4' in current directory. If a path to a directory
        that does not exist, will create the directory path.
    fps: int
        Frames per second. Passed into moviepy->ImageSequenceClip() method
    **kwargs: dict
        Optional keywords passed into moviepy->write_videofile() method


    Returns
    -------
    write_filename : str
        Full path name of created movie file

    """
    if not MOVIEPY_AVAILABLE:
        raise ImportError('MoviePy needs to be installed on your system to make movies.')

    # Set default movie name
    if write_filename is None:
        write_filename = Path(Path().cwd(), 'movie.mp4')

    # Check if images is pointing to a directory. If so ensure is a string not PosixPath
    IS_MOVIE = False
    if isinstance(images, (types.GeneratorType, list, tuple)):
        images = [str(image) for image in images]
        images.sort()
    elif isinstance(images, (PathLike, str)) and Path(images).is_file():
        IS_MOVIE = True
        images = str(images)
    elif isinstance(images, (PathLike, str)) and Path(images).is_dir():
        images = str(images)

    # Ensure full path to filename exists
    write_directory = Path(write_filename).parent
    write_directory.mkdir(parents=True, exist_ok=True)

    if IS_MOVIE:
        with moviepy_editor.VideoFileClip(images) as clip:
            # There can be an issue converting mpeg to other movie format because the
            # duration parameter in the movie file is not set. So moviepy guesses and
            # can get the duration wrong. This will find the correct duration (correct to 0.2 seconds)
            # and set before writing.
            if Path(images).suffix == '.mpg':
                import numpy as np
                import warnings
                from collections import deque

                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    desired_len = 3
                    frame_sums = deque()
                    duration = 0.0  # Duration of movie in seconds
                    while True:
                        result = clip.get_frame(duration)
                        frame_sums.append(np.sum(result))
                        if len(frame_sums) > desired_len:
                            frame_sums.popleft()

                            if len(set(frame_sums)) == 1:
                                break

                        duration += 0.1

                    clip = clip.set_start(0)
                    clip = clip.set_duration(duration)
                    clip = clip.set_end(duration)
                    clip.write_videofile(str(write_filename), **kwargs)

            else:
                clip.write_videofile(str(write_filename), **kwargs)

    else:
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=fps)
        clip.write_videofile(str(write_filename), **kwargs)

    return str(write_filename)


def arm_standards_validator(file=None, dataset=None, verbose=True):
    """
    ARM Data Validator (ADV) - Checks to ensure that ARM standards are being followed
    in the files or dataset passed to it.  Note, this includes a minimal set of
    standards that it checks against

    Parameters
    ----------
    file : str
        Filename to check against ARM standards.  Do not pass in both a file and dataset
    dataset : xarray.DataSet
        Xarray dataset of an already read in file.
    verbose : boolean
        Defaults to print out errors in addition to returning a list of them

    Returns
    -------
    err : list
        List of errors in the data

    """

    # Set up the error tracking list
    err = []
    if file is not None and isinstance(file, str):
        # Check file naming standards
        if len(file.split(sep)[-1]) > 60.0:
            err.append('Filename length exceeds 60 characters')
        try:
            f_obj = act.utils.data_utils.DatastreamParserARM(file)
        except Exception as e:
            print(e)

        if (
            (f_obj.site is None)
            or (f_obj.datastream_class is None)
            or (f_obj.level is None)
            or (f_obj.facility is None)
            or (f_obj.date is None)
            or (f_obj.time is None)
            or (f_obj.ext is None)
        ):
            err.append(
                'Filename does not follow the normal ARM convention: '
                + '(sss)(inst)(qualifier)(temporal)(Fn).(dl).(yyyymmdd).(hhmmss).nc'
            )
        else:
            if f_obj.level[0] not in ['0', 'a', 'b', 'c', 's', 'm']:
                err.append(f_obj.level + ' is not a standard ARM data level')

        results = act.utils.arm_site_location_search(
            site_code=f_obj.site, facility_code=f_obj.facility
        )
        if len(results) == 0:
            err.append('Site and facility are not ARM standard')

    # The ability to read a file from NetCDF to xarray will catch a lot of the
    # problems with formatting.  This would leave standard ARM checks
    try:
        if dataset is None and file is not None:
            ds = act.io.read_arm_netcdf(file)
        elif dataset is not None:
            ds = dataset
        else:
            raise ValueError('File and dataset are both None')
    except Exception as e:
        return ['File is not in a standard format that is readable by xarray: ' + str(e)]

    # Review time variables for errors for conformance to standards
    if 'time' not in list(ds.dims)[0]:
        err.append('"time" is required to be the first dimension.')

    for c in list(ds.coords):
        if c not in ds.dims:
            err.append(c + ': Coordinate is not included in dimensions.')

    if any(np.isnan(ds['time'].values)):
        err.append('Time must not include NaNs.')

    duplicates = sum(ds['time'].to_pandas().duplicated())
    if duplicates > 0:
        err.append('Duplicate times present in the file')

    diff = ds['time'].diff('time')
    idx = np.where(diff <= pd.Timedelta(0))
    if len(idx[0]) > 0:
        err.append('Time is not in increasing order')

    if 'base_time' not in ds or 'time_offset' not in ds:
        err.append('ARM requires base_time and time_offset variables.')

    # Check to make sure other coordinate variables don't have nans
    # Also check to make sure coordinate variables are not decreasing
    if len(list(ds.coords)) > 1:
        for d in ds.coords:
            if d == 'time':
                continue
            if any(np.isnan(ds[d].values)):
                err.append('Coordinates must not include NaNs ' + d)

            diff = ds[d].diff(d)
            idx = np.where(diff <= 0.0)
            if len(idx[0]) > 0:
                err.append(d + ' is not in increasing order')
            if 'missing_value' in ds[d].encoding:
                err.append(d + ' should not include missing value')

    # Verify that each variable has a long_name and units attributes
    for v in ds:
        if (len(ds[v].dims) > 0) and ('time' not in list(ds[v].dims)[0]) and ('bounds' not in v):
            err.append(v + ': "time" is required to be the first dimension.')
        if (ds[v].size == 1) and (len(ds[v].dims) > 0):
            err.append(v + ': is not defined as a scalar.')
        if 'long_name' not in ds[v].attrs:
            err.append('Required attribute long_name not in ' + v)
        else:
            if not ds[v].attrs['long_name'][0].isupper():
                err.append(v + ' long_name attribute does not start with uppercase')

        if (
            ('qc_' not in v)
            and (v not in ['time', 'time_offset', 'base_time', 'lat', 'lon', 'alt'])
            and ('bounds' not in v)
        ):
            if ('missing_value' not in ds[v].encoding) and ('FillValue' not in ds[v].encoding):
                err.append(v + ' does not include missing_value or FillValue attributes')

        # QC variable checks
        if 'qc_' in v:
            if v[3:] not in ds:
                err.append('QC variable does not have a corresponding variable ' + v[3:])
            if 'ancillary_variables' not in ds[v[3:]].attrs:
                err.append(
                    v[3:] + ' does not include ancillary_variable attribute pointing to ' + v
                )
            if 'description' not in ds[v].attrs:
                err.append(v + ' does not include description attribute')
            if 'flag_method' not in ds[v].attrs:
                err.append(v + ' does not include flag_method attribute')

        if (v not in ['base_time', 'time_offset']) and ('bounds' not in v):
            if 'units' not in ds[v].attrs:
                err.append('Required attribute units not in ' + v)

    # Lat/Lon/Alt Checks
    if 'lat' not in ds:
        err.append('ARM requires the latitude variable to be named lat')
    else:
        if 'standard_name' in ds['lat'].attrs:
            if ds['lat'].attrs['standard_name'] != 'latitude':
                err.append('ARM requires the lat standard_name to be latitude')
        else:
            err.append('"lat" variable does not have a standard_name attribute')
    if 'lon' not in ds:
        err.append('ARM requires the longitude variable to be named lon')
    else:
        if 'standard_name' in ds['lon'].attrs:
            if ds['lon'].attrs['standard_name'] != 'longitude':
                err.append('ARM requires the lon standard_name to be longitude')
        else:
            err.append('"long" variable does not have a standard_name attribute')
    if 'alt' not in ds:
        err.append('ARM requires the altitude variable to be named alt')
    else:
        if 'standard_name' in ds['alt'].attrs:
            if ds['alt'].attrs['standard_name'] != 'altitude':
                err.append('ARM requires the alt standard_name to be altitude')
        else:
            err.append('"alt" variable does not have a standard_name attribute')

    # Required global attributes
    req_att = ['doi', 'sampling_interval', 'averaging_interval']
    for ra in req_att:
        if ra not in ds.attrs:
            err.append('Global attribute is missing: ' + ra)

    if verbose:
        if len(err) > 0:
            [print(e) for e in err]
        else:
            print('File is passing standards checks')

    return err
