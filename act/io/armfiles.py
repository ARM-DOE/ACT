"""
===============
act.io.armfiles
===============

This module contains I/O operations for loading files that were created for the
Atmospheric Radiation Measurement program supported by the Department of Energy
Office of Science.

"""
# import standard modules
import glob
import xarray as xr
import numpy as np
# import warnings

# from .dataset import ACTAccessor
from enum import Flag, auto


class ARMStandardsFlag(Flag):
    """
    This class stores a flag that is returned by
    :ref:act.io.armfiles.check_arm_standards.

    Attributes
    ----------
    OK : flag
        This flag is set if the dataset conforms to ARM standards.
    NO_DATASTREAM : flag
        This flag is set if the dataset does not have a datastream
        field.

    Examples
    --------
    .. code-block:: python

         my_flag = act.io.armfiles.ARMStandardsFlag(
             act.io.armfiles.ARMStandardsFlag.OK)
         assert my_flag.OK

    """
    OK = auto()
    """ The dataset conforms to ARM standards. """
    NO_DATASTREAM = auto()
    """ The dataset does not have a datastream field. """


def read_netcdf(filenames, concat_dim='time', return_None=False, **kwargs):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ARM-standard netCDF files from a single datastream.

    Parameters
    ----------
    filenames : str or list
        Name of file(s) to read.
    concat_dim : str
        Dimension to concatenate files along. Default value is 'time.'
    return_none : bool, optional
        Catch IOError exception when file not found and return None.
        Default is False.
    **kwargs : keywords
        Keywords to pass through to xarray.open_mfdataset().

    Returns
    -------
    act_obj : Object (or None)
        ACT dataset (or None if no data file(s) found).

    Examples
    --------
    This example will load the example sounding data used for unit testing.

    .. code-block:: python

        import act

        the_ds, the_flag = act.io.armfiles.read_netcdf(
            act.tests.sample_files.EXAMPLE_SONDE_WILDCARD)
        print(the_ds.act.datastream)

    """
    file_dates = []
    file_times = []
    if return_None:
        try:
            arm_ds = xr.open_mfdataset(filenames, combine='nested',
                                       concat_dim=concat_dim, **kwargs)
        except IOError:
            return None
    else:
        arm_ds = xr.open_mfdataset(filenames, combine='nested',
                                   concat_dim=concat_dim, **kwargs)

    # Check if time variable is not in the netCDF file and if so look
    # to use time_offset to make time dimension.
    try:
        if not np.issubdtype(type(arm_ds['time'].values[0]), np.datetime64):
            arm_ds.rename({'time_offset': 'time'}, inplace=True)
            arm_ds.set_coords('time', inplace=True)
    except (KeyError, ValueError):
        pass

    # Adding support for wildcards
    if isinstance(filenames, str):
        filenames = glob.glob(filenames)

    filenames.sort()
    for n, f in enumerate(filenames):
        file_dates.append(f.split('.')[-3])
        file_times.append(f.split('.')[-2])

    arm_ds.act.file_dates = file_dates
    arm_ds.act.file_times = file_times
    is_arm_file_flag = check_arm_standards(arm_ds)

    if is_arm_file_flag.NO_DATASTREAM is True:
        arm_ds.act.datastream = "act_datastream"
    else:
        arm_ds.act.datastream = arm_ds.attrs["datastream"]
    arm_ds.act.site = str(arm_ds.act.datastream)[0:3]
    arm_ds.act.arm_standards_flag = is_arm_file_flag

    return arm_ds


def check_arm_standards(ds):
    """
    Checks to see if an xarray dataset conforms to ARM standards.

    Parameters
    ----------
    ds : xarray dataset
        The dataset to check.

    Returns
    -------
    flag : ARMStandardsFlag
        The flag corresponding to whether or not the file conforms
        to ARM standards.

    """
    the_flag = ARMStandardsFlag(ARMStandardsFlag.OK)
    the_flag.NO_DATASTREAM = False
    the_flag.OK = True
    if 'datastream' not in ds.attrs.keys():
        the_flag.OK = False
        the_flag.NO_DATASTREAM = True

    return the_flag
