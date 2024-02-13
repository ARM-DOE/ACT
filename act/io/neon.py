"""
Modules for reading in NOAA PSL data.
"""


import pandas as pd
import xarray as xr

from .text import read_csv


def read_neon_csv(files, variable_files=None, position_files=None):
    """
    Reads in the NEON formatted csv files from local paths or urls
    and returns an Xarray dataset.

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
    ds : xarray.Dataset
        Standard Xarray dataset

    """

    # Raise error if empty list is passed in
    if len(files) == 0:
        raise ValueError('File list is empty')
    if isinstance(files, str):
        files = [files]

    # Read in optional files
    multi_ds = []
    if variable_files is not None:
        if isinstance(variable_files, str):
            variable_files = [variable_files]
        df = pd.read_csv(variable_files[0])

    if position_files is not None:
        if isinstance(position_files, str):
            position_files = [position_files]
        loc_df = pd.read_csv(position_files[0], dtype=str)

    # Run through each file and read into a dataset
    for i, f in enumerate(files):
        ds = read_csv(f)
        # Create standard time variable
        time = [pd.to_datetime(t).replace(tzinfo=None) for t in ds['startDateTime'].values]
        ds['time'] = xr.DataArray(data=time, dims=['index'])
        ds['time'].attrs['units'] = ''
        ds = ds.swap_dims({'index': 'time'})
        ds = ds.drop_vars('index')

        # Add some metadata
        site_code = f.split('/')[-1].split('.')[2]
        resolution = f.split('/')[-1].split('.')[9]
        hor_loc = f.split('/')[-1].split('.')[6]
        ver_loc = f.split('/')[-1].split('.')[7]
        ds.attrs['_sites'] = site_code
        ds.attrs['averaging_interval'] = resolution.split('_')[-1]
        ds.attrs['HOR.VER'] = hor_loc + '.' + ver_loc

        # Add in metadata from the variables file
        if variable_files is not None:
            for v in ds:
                dummy = df.loc[(df['table'] == resolution) & (df['fieldName'] == v)]
                ds[v].attrs['units'] = str(dummy['units'].values[0])
                ds[v].attrs['long_name'] = str(dummy['description'].values[0])
                ds[v].attrs['format'] = str(dummy['pubFormat'].values[0])

        # Add in sensor position data
        if position_files is not None:
            dloc = loc_df.loc[loc_df['HOR.VER'] == hor_loc + '.' + ver_loc]
            idx = dloc.index.values
            if len(idx) > 0:
                ds['lat'] = xr.DataArray(data=float(loc_df['referenceLatitude'].values[idx]))
                ds['lon'] = xr.DataArray(data=float(loc_df['referenceLongitude'].values[idx]))
                ds['alt'] = xr.DataArray(data=float(loc_df['referenceElevation'].values[idx]))
                variables = [
                    'xOffset',
                    'yOffset',
                    'zOffset',
                    'eastOffset',
                    'northOffset',
                    'pitch',
                    'roll',
                    'azimuth',
                    'xAzimuth',
                    'yAzimuth',
                ]
                for v in variables:
                    ds[v] = xr.DataArray(data=float(loc_df[v].values[idx]))
        multi_ds.append(ds)

    ds = xr.merge(multi_ds)

    return ds
