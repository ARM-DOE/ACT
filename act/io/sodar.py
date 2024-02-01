"""
This module contains I/O operations for loading Sodar files.

"""

import datetime as dt
import re

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from act.io.noaapsl import filter_list


def read_mfas_sodar(filepath):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    Flat Array MFAS Sodar file. More information can be found here:
    https://www.scintec.com/products/flat-array-sodar-mfas/

    Parameters
    ----------
    filepath : str
        Name of file to read.

    Return
    ------
    ds : xarray.Dataset
        Standard Xarray dataset with the data.

    """
    file = fsspec.open(filepath).open()
    lines = file.readlines()
    lines = [x.decode().rstrip()[:] for x in lines]

    # Retrieve number of height values from line 3.
    _, _, len_height = filter_list(lines[3].split()).astype(int)

    # Retrieve metadata
    file_dict, variable_dict = _metadata_retrieval(lines)

    # Retrieve datetimes and time indices from when datetime rows appear
    skip_time_ind = []
    datetimes = []
    fmt = '%Y-%m-%d %H:%M:%S'
    for i, line in enumerate(lines):
        match = re.search(r'\d{4}-\d{2}-\d{2}\ \d{2}:\d{2}:\d{2}', line)
        if match is None:
            continue
        else:
            date_object = dt.datetime.strptime(match.group(0), fmt)
            datetimes.append(date_object)
            skip_time_ind.append(i)

    datetimes = np.delete(datetimes, 0)
    # Create datetime column with matching datetimes to heights
    data_times = pd.DataFrame(datetimes, columns=['Dates'])
    repeat_times = data_times.loc[data_times.index.repeat(len_height)]

    # This is used to pull only actual data.
    # Code can be added as well to read in the metadata from the first few rows.
    skip_meta_ind = np.arange(0, skip_time_ind[1] + 1, 1)
    skip_full_ind = np.append(skip_meta_ind, skip_time_ind)
    skip_full_ind = np.unique(skip_full_ind)

    # Column row appears 1 row after first time, retrieve column names from that.
    columns = np.delete(filter_list(lines[skip_time_ind[1] + 1].split(' ')), 0).tolist()

    # Tmp column allows for the # column to be pushed over and dropped.
    tmp_columns = columns + ['tmp']

    # Parse data to a dataframe skipping rows that aren't data.
    # tmp_columns is used to removed '#' column that causes
    # columns to move over by one.
    df = pd.read_table(
        filepath, sep=r'\s+', skiprows=skip_full_ind, names=tmp_columns, usecols=columns
    )

    df = df[~df['W'].isin(['dir'])].reset_index(drop=True)

    # Set index to datetime column.
    df = df.set_index(repeat_times['Dates'])

    # Convert dataframe to xarray dataset.
    ds = df.to_xarray()

    # Convert height to float.
    ds['z'] = ds.z.astype(float)

    # Convert all variables from string to float.
    ds = ds.astype(float)

    # Convert variables that should be int back to int.
    ds['error'] = ds.error.astype(int)
    ds['PGz'] = ds.PGz.astype(int)

    # Get unique time and height values.
    time_dim = np.unique(ds.Dates.values)
    height_dim = np.unique(ds.z.values)

    # Use unique time and height values to reindex data to be two dimensional.
    ind = pd.MultiIndex.from_product((time_dim, height_dim), names=('time', 'height'))

    # Xarray 2023.9 contains new syntax, adding try and except for
    # previous version.
    try:
        mindex_coords = xr.Coordinates.from_pandas_multiindex(ind, 'Dates')
        ds = ds.assign_coords(mindex_coords).unstack("Dates")
    except AttributeError:
        ds = ds.assign(Dates=ind).unstack("Dates")

    # Add file metadata.
    for key in file_dict.keys():
        ds.attrs[key] = file_dict[key]

    # Add metadata to the attributes of each variable.
    for key in variable_dict.keys():
        ds[key].attrs = variable_dict[key]

    # Change fill values to nans for floats and 0 for ints.
    # We can't use xr.replace as the fill value changes between variables.
    for var in ds.data_vars:
        if var == 'error':
            continue
        elif var == 'PGz':
            data_with_fill = ds[var].values
            data_with_fill[data_with_fill == 99] = 0
            ds[var].values = data_with_fill
        else:
            data_with_fill = ds[var].values
            fill_value = ds[var].attrs['_FillValue']
            data_with_fill[data_with_fill == fill_value] = np.nan
            ds[var].values = data_with_fill

    # Drop z as its already a coordinate and give coordinate the same attributes.
    ds.height.attrs = ds['z'].attrs
    ds = ds.drop_vars('z')

    return ds


def _metadata_retrieval(lines):
    # File format from line 0.
    _format = lines[0]

    # Sodar type from line 2.
    instrument_type = lines[2]

    # Create np.array of lines to use np.argwhere
    line_array = np.array(lines)

    # Retrieve indices of file information and the end of the metadata block.
    file_info_ind = np.argwhere(line_array == '# file information')[0][0]
    file_type_ind = np.argwhere(line_array == '# file type')[0][0]

    # Index the section of file information.
    file_def = line_array[file_info_ind + 2 : file_type_ind - 1]

    # Create a dictionary of file information to be plugged in later to the xarray
    # dataset attributes.
    file_dict = {}
    for line in file_def:
        key, value = filter_list(line.split(':'))
        file_dict[key.strip()] = value.strip()
    file_dict['format'] = _format
    file_dict['instrument_type'] = instrument_type

    # Change values from strings to float where need be.
    file_dict['antenna azimuth angle [deg]'] = float(file_dict['antenna azimuth angle [deg]'])
    file_dict['height above ground [m]'] = float(file_dict['height above ground [m]'])
    file_dict['height above sea level [m]'] = float(file_dict['height above sea level [m]'])

    # Retrieve indices of variable information.
    variable_info_ind = np.argwhere(line_array == '# variable definitions')[0][0]
    data_ind = np.argwhere(line_array == '# beginning of data block')[0][0]

    # Index the section of variable information.
    variable_def = line_array[variable_info_ind + 2 : data_ind - 1]

    # Create a dictionary of variable information to be plugged in later to the xarray
    # variable attributes. Skipping error code as it does not have metadata similar to
    # the rest of the variables.
    variable_dict = {}
    for i, line in enumerate(variable_def):
        if 'error code' in line:
            continue
        else:
            temp_var_dict = {}
            key, symbol, units, _type, error_mask, fill_value = filter_list(line.split('#'))
            temp_var_dict['variable_name'] = key.strip()
            temp_var_dict['symbol'] = symbol.strip()
            temp_var_dict['units'] = units.strip()
            temp_var_dict['type'] = _type.strip()
            temp_var_dict['error_mask'] = error_mask.strip()
            if key.strip() == 'PGz':
                temp_var_dict['_FillValue'] = int(fill_value)
            else:
                temp_var_dict['_FillValue'] = float(fill_value)
            variable_dict[symbol.strip()] = temp_var_dict

    return file_dict, variable_dict
