"""
Modules for reading in NOAA PSL data.
"""

import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
from act.io.csvfiles import read_csv


def read_neon_csv(files, variable_files=None, position_files=None):
    """
    Reads in the NEON formatted csv files from local paths or urls
    and returns an xarray dataset.

    Parameters
    ----------
    filepath : list
        Files to read in
    variable_files : list
        Name of variable files to read with metadata. Optional but the Dataset will not
        have any metadata
    position_files : list
        Name of file to read with sensor positions.  Optional, but the Dataset will not
        have any location information

    Return
    ------
    obj : Xarray.dataset
        Standard Xarray dataset

    """

    # Raise error if empty list is passed in
    if len(files) == 0:
        raise ValueError('File list is empty')
    if isinstance(files, str):
        files = [files]

    # Read in optional files
    objs = []
    if variable_files is not None:
        if isinstance(variable_files, str):
            variable_files = [variable_files]
        df = pd.read_csv(variable_files[0])

    if position_files is not None:
        if isinstance(position_files, str):
            position_files = [position_files]
        loc_df = pd.read_csv(position_files[0], dtype=str)

    # Run through each file and read into an object
    for i, f in enumerate(files):
        obj = read_csv(f)
        # Create standard time variable
        time = [pd.to_datetime(t).replace(tzinfo=None) for t in obj['startDateTime'].values]
        obj['time'] = xr.DataArray(data=time, dims=['index'])
        obj['time'].attrs['units'] = ''
        obj = obj.swap_dims({'index': 'time'})
        obj = obj.drop_vars('index')

        # Add some metadata
        site_code = f.split('/')[-1].split('.')[2]
        resolution = f.split('/')[-1].split('.')[9]
        hor_loc = f.split('/')[-1].split('.')[6]
        ver_loc = f.split('/')[-1].split('.')[7]
        obj.attrs['_sites'] = site_code
        obj.attrs['averaging_interval'] = resolution.split('_')[-1]
        obj.attrs['HOR.VER'] = hor_loc + '.' + ver_loc

        # Add in metadata from the variables file
        if variable_files is not None:
            for v in obj:
                dummy = df.loc[(df['table'] == resolution) & (df['fieldName'] == v)]
                obj[v].attrs['units'] = str(dummy['units'].values[0])
                obj[v].attrs['long_name'] = str(dummy['description'].values[0])
                obj[v].attrs['format'] = str(dummy['pubFormat'].values[0])

        # Add in sensor position data
        if position_files is not None:
            dloc = loc_df.loc[loc_df['HOR.VER'] == hor_loc + '.' + ver_loc]
            idx = dloc.index.values
            if len(idx) > 0:
                obj['lat'] = xr.DataArray(data=float(loc_df['referenceLatitude'].values[idx]))
                obj['lon'] = xr.DataArray(data=float(loc_df['referenceLongitude'].values[idx]))
                obj['alt'] = xr.DataArray(data=float(loc_df['referenceElevation'].values[idx]))
                variables = ['xOffset', 'yOffset', 'zOffset', 'eastOffset', 'northOffset',
                             'pitch', 'roll', 'azimuth', 'xAzimuth', 'yAzimuth']
                for v in variables:
                    obj[v] = xr.DataArray(data=float(loc_df[v].values[idx]))
        objs.append(obj)

    obj = xr.merge(objs)

    return obj
