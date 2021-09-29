"""
Modules for reading in NOAA PSL data.
"""

import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr


def read_psl_wind_profiler(filename, transpose=True):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    NOAA PSL wind profiler file.

    Parameters
    ----------
    filename : str
        Name of file(s) to read.
    transpose : bool
        True to transpose the data.

    Return
    ------
    obj_low :  Xarray.dataset
        Standard Xarray dataset with the data for low mode
    obj_high : Xarray.dataset
        Standard Xarray dataset with the data for high mode.

    """
    # read file with pandas for preparation.
    df = pd.read_csv(filename, header=None)

    # Get location of where each table begins
    index_list = df[0] == ' CTD'
    idx = np.where(index_list)

    # Get header of each column of data.
    column_list = list(df.loc[9][0].split())

    beam_vars = ['RAD', 'CNT', 'SNR', 'QC']
    for i, c in enumerate(column_list):
        if c in beam_vars:
            if column_list.count(c) > 2:
                column_list[i] = c + '1'
            elif column_list.count(c) > 1:
                column_list[i] = c + '2'
            elif column_list.count(c) > 0:
                column_list[i] = c + '3'

    # Loop through column data only which appears after 10 lines of metadata.
    # Year, Month, day, hour, minute, second, utc offset
    low = []
    hi = []
    for i in range(idx[0].shape[0] - 1):
        # index each table by using the idx of when CTD appears.
        # str split is use as 2 spaces are added to each data point,
        # convert to float.
        date_str = df.iloc[idx[0][i] + 3]
        date_str = list(filter(None, date_str[0].split(' ')))
        date_str = list(map(int, date_str))
        # Datetime not taking into account the utc offset yet
        time = dt.datetime(
            2000 + date_str[0], date_str[1], date_str[2], date_str[3],
            date_str[4], date_str[5])

        mode = df.iloc[idx[0][i] + 7][0]
        mode = int(mode.split(' ')[-1])

        df_array = np.array(
            df.iloc[idx[0][i] + 10:idx[0][i + 1] - 1][0].str.split(
                '\s{2,}').tolist(), dtype='float')
        df_add = pd.DataFrame(df_array, columns=column_list)
        df_add = df_add.replace(999999.0, np.nan)

        xr_add = df_add.to_xarray()
        xr_add = xr_add.swap_dims({'index': 'height'})
        xr_add = xr_add.reset_coords('index')
        xr_add = xr_add.assign_coords(
            {'time': np.array(time), 'height': xr_add['HT'].values})

        if mode < 1000.:
            low.append(xr_add)
        else:
            hi.append(xr_add)

    obj_low = xr.concat(low, 'time')
    obj_hi = xr.concat(hi, 'time')

    # Adding site information line 1
    site_loc = df.iloc[idx[0][0]]
    site_list = site_loc.str.split('\s{2}').tolist()
    site = site_list[0][0].strip()

    obj_low.attrs['site_identifier'] = site
    obj_hi.attrs['site_identifier'] = site

    # Adding data type and revision number line 2.
    rev = df.loc[idx[0][0] + 1]
    rev_list = rev.str.split('\s{3}').tolist()
    rev_array = np.array(rev_list[0])

    obj_low.attrs['data_type'] = rev_array[0].strip()
    obj_hi.attrs['data_type'] = rev_array[0].strip()
    obj_low.attrs['revision_number'] = rev_array[1].strip()
    obj_hi.attrs['revision_number'] = rev_array[1].strip()

    # Adding coordinate attributes line 3.
    coords = df.loc[idx[0][0] + 2]
    coords_list = coords.str.split('\s{2,}').tolist()
    coords_list[0].remove('')
    coords_array = np.array(coords_list[0], dtype='float32')

    obj_low.attrs['latitude'] = np.array([coords_array[0]])
    obj_hi.attrs['latitude'] = np.array([coords_array[0]])
    obj_low.attrs['longitude'] = np.array([coords_array[1]])
    obj_hi.attrs['longitude'] = np.array([coords_array[1]])
    obj_low.attrs['altitude'] = np.array([coords_array[2]])
    obj_hi.attrs['altitude'] = np.array([coords_array[2]])

    # Adding azimuth and elevation line 9
    az_el = df.loc[idx[0][0] + 8]
    az_el_list = az_el.str.split('\s{2,}').tolist()
    az_el_list[0].remove('')
    az_el_array = np.array(az_el_list[0])
    az = []
    el = []
    for i in az_el_array:
        sep = i.split()
        az.append(sep[0])
        el.append(sep[1])
    az_array = np.array(az, dtype='float32')
    el_array = np.array(el, dtype='float32')

    obj_low.attrs['azimuth'] = az_array
    obj_hi.attrs['azimuth'] = az_array
    obj_low.attrs['elevation'] = el_array
    obj_hi.attrs['elevation'] = el_array

    if transpose:
        obj_low = obj_low.transpose()
        obj_hi = obj_hi.transpose()

    return obj_low, obj_hi
