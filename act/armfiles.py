"""
armit.io.read_data_obj
==================
Reads ARM NetCDF files and store in standard object
.. autosummary::
    :toctree: generated/
    read
    determine_filetype
"""

# import standard modules
import os
import re
import datetime
import calendar
import time
from pathlib import Path
from collections import OrderedDict
import filecmp

# import 3rd-party modules
import numpy as np
import netCDF4
from pandas import DataFrame
import xarray as xr
from matplotlib.dates import date2num
from dateutil.parser import parse as dateparse

def read_netcdf(filenames, variables=None):

    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ARM-standard netCDF files from a single datastream.

    Parameters
    ----------
    filenames : str or list
        Name of file(s) to read

    Other Parameters
    ----------
    variables : list, optional
        List of variable name(s) to read

    Returns
    ----------
    arm_obj : Object
        Xarray data object

    """

    for n, f in enumerate(filenames):
        try:
            ds = xr.open_dataset(f)
        except: 
            continue
       
        if n == 0:
            arm_ds = ds
        else:
            arm_ds = xr.concat([arm_ds,ds],dim='time')

    return arm_ds 
