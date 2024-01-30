"""
This module contains I/O operations for loading MPL files.

"""

import os
import shutil
import subprocess
import tempfile
import dask
import xarray as xr
from act.io.arm import check_arm_standards

if shutil.which('mpl2nc') is not None:
    MPLIMPORT = True
else:
    MPLIMPORT = False


def read_sigma_mplv5(
    filename,
    save_nc=False,
    out_nc_path=None,
    afterpulse=None,
    dead_time=None,
    overlap=None,
    **kwargs,
):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    SIGMA MPL V5 files. File is converted to netCDF using mpl2nc an optional
    dependency.

    Parameters
    ----------
    filename : str
        Name of file(s) to read.  If multiple, ensure that they are sorted,
        otherwise they will not concat properly
    save_nc : bool
        Whether or not to save intermediate step nc file.
    out_nc_path : str
        Path to save intermediate step nc file.
    afterpulse : str
        File with afterpulse correction (.bin)
    dead_time : str
        File with dead time correction (.bin)
    overlap : str
        File with overlap correction (.bin)
    **kwargs : keywords
        Keywords to pass through to xarray.open_dataset().

    """
    if not MPLIMPORT:
        raise ImportError(
            'The module mpl2nc is not installed and is needed to read ' 'mpl binary files!'
        )

    if isinstance(filename, str):
        filename = [filename]

    task = []
    for f in filename:
        task.append(
            dask.delayed(proc_sigma_mplv5_read)(
                f,
                save_nc=save_nc,
                out_nc_path=out_nc_path,
                afterpulse=afterpulse,
                dead_time=dead_time,
                overlap=overlap,
                **kwargs,
            )
        )

    results_ds = dask.compute(*task)

    ds = xr.concat(results_ds, 'time')

    return ds


def proc_sigma_mplv5_read(
    f, save_nc=False, out_nc_path=None, afterpulse=None, dead_time=None, overlap=None, **kwargs
):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    SIGMA MPL V5 files. File is converted to netCDF using mpl2nc an optional
    dependency.  Dask sub-routine for processing

    Parameters
    ----------
    filename : str
        Name of file to read.
    save_nc : bool
        Whether or not to save intermediate step nc file.
    out_nc_path : str
        Path to save intermediate step nc file.
    afterpulse : str
        File with afterpulse correction (.bin)
    dead_time : str
        File with dead time correction (.bin)
    overlap : str
        File with overlap correction (.bin)
    **kwargs : keywords
        Keywords to pass through to xarray.open_dataset().

    """

    datastream_name = '.'.join(f.split('/')[-1].split('.')[0:2])
    if '.bin' not in f:
        mpl = True
        tmpfile1 = tempfile.mkstemp(suffix='.bin', dir='.')[1]
        shutil.copyfile(f, tmpfile1)
        f = tmpfile1
    else:
        mpl = False

    # Set up call for mpl2nc to add in correction files
    call = 'mpl2nc'
    if afterpulse is not None:
        call += ' -a ' + afterpulse
    if dead_time is not None:
        call += ' -d ' + dead_time
    if overlap is not None:
        call += ' -o ' + overlap

    call += ' ' + f

    # Specify the output, will use a temporary file if no output specified
    if save_nc:
        if out_nc_path is None:
            raise ValueError('You are using save_nc, please specify ' 'an out_nc_path')

        subprocess.call(call + ' ' + out_nc_path, shell=True)
        ds = xr.open_dataset(out_nc_path, **kwargs)
    else:
        tmpfile2 = tempfile.mkstemp(suffix='.nc', dir='.')[1]
        subprocess.call(call + ' ' + tmpfile2, shell=True)
        ds = xr.open_dataset(tmpfile2, **kwargs)
        os.remove(tmpfile2)

    # Calculate range in meters
    ds['range'] = 0.5 * ds.bin_time[0] * ds.c * (ds.range + 0.5)

    # Swap the coordinates to be time and range
    ds = ds.swap_dims({'profile': 'time'})
    ds = ds.assign_coords({'time': ds.time, 'range': ds.range})

    # Add metadata
    is_arm_file_flag = check_arm_standards(ds)
    if is_arm_file_flag == 0:
        ds.attrs['_datastream'] = datastream_name
    ds.attrs['_arm_standards_flag'] = is_arm_file_flag

    if mpl:
        os.remove(tmpfile1)

    return ds
