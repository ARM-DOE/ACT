"""
==========
act.io.mpl
==========

This module contains I/O operations for loading MPL files.

"""

import os
import shutil
import subprocess
import tempfile

import pandas
import xarray as xr

from act.io.armfiles import check_arm_standards

try:
    subprocess.call('mpl2nc')
    MPLIMPORT = True
except FileNotFoundError:
    MPLIMPORT = False


def read_sigma_mplv5(filename, save_nc=False, out_nc_path=None,
                     **kwargs):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    SIGMA MPL V5 files. File is converted to netCDF using mpl2nc an optional
    dependency.
    
    Parameters
    ----------
    filename : str
        Name of file to read.
    save_nc : bool
        Whether or not to save intermediate step nc file.
    out_nc_path : str
        Path to save intermediate step nc file.
    **kwargs : keywords
        Keywords to pass through to xarray.open_dataset().

    """
    if not MPLIMPORT:
        raise ImportError(
            'The modulempl2nc is not installed and is needed to read '
            'mpl binary files!')

    if '.bin' not in filename:
        mpl = True
        tmpfile1 = tempfile.mkstemp(suffix='.bin', dir='.')[1]
        shutil.copyfile(filename, tmpfile1)
        filename = tmpfile1
    else:
        mpl = False

    if save_nc:
        if out_nc_path is None:
            raise ValueError(
                'You are using save_nc, please specify '
                'an out_nc_path')

        subprocess.call('mpl2nc ' + filename + ' ' + out_nc_path, shell=True)
        ds = xr.open_dataset(out_nc_path, **kwargs)
    else:
        tmpfile2 = tempfile.mkstemp(suffix='.nc', dir='.')[1]
        subprocess.call('mpl2nc ' + filename + ' ' + tmpfile2, shell=True)
        ds = xr.open_dataset(tmpfile2, **kwargs)
        os.remove(tmpfile2)

    ds = ds.assign_coords({'profile': ds.profile,
                           'range': ds.range})

    is_arm_file_flag = check_arm_standards(ds)
    if is_arm_file_flag.NO_DATASTREAM is True:
        ds.attrs['_datastream'] = '.'.join(
            filename.split('/')[-1].split('.')[0:2])
    ds.attrs['_arm_standards_flag'] = is_arm_file_flag

    if mpl:
        os.remove(tmpfile1)
    return ds
