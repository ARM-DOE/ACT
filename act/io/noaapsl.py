"""
Modules for reading in NOAA PSL data.
"""

from datetime import datetime, timedelta
from os import path as ospath
import fsspec
import yaml
import re
import numpy as np
import pandas as pd
import xarray as xr

from act.io.csvfiles import read_csv


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
        time = datetime(
            2000 + date_str[0],
            date_str[1],
            date_str[2],
            date_str[3],
            date_str[4],
            date_str[5],
        )

        mode = df.iloc[idx[0][i] + 7][0]
        mode = int(mode.split(' ')[-1])

        df_array = np.array(
            df.iloc[idx[0][i] + 10 : idx[0][i + 1] - 1][0].str.split(r'\s{2,}').tolist(),
            dtype='float',
        )
        df_add = pd.DataFrame(df_array, columns=column_list)
        df_add = df_add.replace(999999.0, np.nan)

        xr_add = df_add.to_xarray()
        xr_add = xr_add.swap_dims({'index': 'height'})
        xr_add = xr_add.reset_coords('index')
        xr_add = xr_add.assign_coords({'time': np.array(time), 'height': xr_add['HT'].values})

        if mode < 1000.0:
            low.append(xr_add)
        else:
            hi.append(xr_add)

    obj_low = xr.concat(low, 'time')
    obj_hi = xr.concat(hi, 'time')

    # Adding site information line 1
    site_loc = df.iloc[idx[0][0]]
    site_list = site_loc.str.split(r'\s{2}').tolist()
    site = site_list[0][0].strip()

    obj_low.attrs['site_identifier'] = site
    obj_hi.attrs['site_identifier'] = site

    # Adding data type and revision number line 2.
    rev = df.loc[idx[0][0] + 1]
    rev_list = rev.str.split(r'\s{3}').tolist()
    rev_array = np.array(rev_list[0])

    obj_low.attrs['data_type'] = rev_array[0].strip()
    obj_hi.attrs['data_type'] = rev_array[0].strip()
    obj_low.attrs['revision_number'] = rev_array[1].strip()
    obj_hi.attrs['revision_number'] = rev_array[1].strip()

    # Adding coordinate attributes line 3.
    coords = df.loc[idx[0][0] + 2]
    coords_list = coords.str.split(r'\s{2,}').tolist()
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
    az_el_list = az_el.str.split(r'\s{2,}').tolist()
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


def read_psl_wind_profiler_temperature(filepath):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    NOAA PSL wind profiler temperature file.

    Parameters
    ----------
    filepath : str
        Name of file(s) to read.

    Return
    ------
    ds :  Xarray.dataset
        Standard Xarray dataset with the data

    """

    # Open the file, read in the lines as a list, and return that list
    file = fsspec.open(filepath).open()
    lines = file.readlines()
    newlist = [x.decode().rstrip()[1:] for x in lines][1:]

    # 1 - site
    site = newlist[0]

    # 2 - datetype
    datatype, _, version = filter_list(newlist[1].split(' '))

    # 3 - station lat, lon, elevation
    latitude, longitude, elevation = filter_list(newlist[2].split('  ')).astype(float)

    # 4 - year, month, day, hour, minute, second, utc
    time = parse_date_line(newlist[3])

    # 5 - Consensus averaging time, number of beams, number of range gates
    consensus_average_time, number_of_beams, number_of_range_gates = filter_list(
        newlist[4].split('  ')
    ).astype(int)

    # 7 - number of coherent integrations, number of spectral averages, pulse width, indder pulse period
    (
        number_coherent_integrations,
        number_spectral_averages,
        pulse_width,
        inner_pulse_period,
    ) = filter_list(newlist[6].split(' ')).astype(int)

    # 8 - full-scale doppler value, delay to first gate, number of gates, spacing of gates
    full_scale_doppler, delay_first_gate, number_of_gates, spacing_of_gates = filter_list(
        newlist[7].split(' ')
    ).astype(float)

    # 9 - beam azimuth (degrees clockwise from north)
    beam_azimuth, beam_elevation = filter_list(newlist[8].split(' ')).astype(float)

    # Read in the data table section using pandas
    df = pd.read_csv(filepath, skiprows=10, delim_whitespace=True)

    # Only read in the number of rows for a given set of gates
    df = df.iloc[: int(number_of_gates)]

    # Nan values are encoded as 999999 - let's reflect that
    df = df.replace(999999.0, np.nan)

    # Ensure the height array is stored as a float
    df['HT'] = df.HT.astype(float)

    # Set the height as an index
    df = df.set_index('HT')

    # Rename the count and snr columns more usefully
    df = df.rename(
        columns={
            'CNT': 'CNT_T',
            'CNT.1': 'CNT_Tc',
            'CNT.2': 'CNT_W',
            'SNR': 'SNR_T',
            'SNR.1': 'SNR_Tc',
            'SNR.2': 'SNR_W',
        }
    )

    # Convert to an xaray dataset
    ds = df.to_xarray()

    # Add attributes to variables
    # Height
    ds['HT'].attrs['long_name'] = 'height_above_ground'
    ds['HT'].attrs['units'] = 'km'

    # Temperature
    ds['T'].attrs['long_name'] = 'average_uncorrected_RASS_temperature'
    ds['T'].attrs['units'] = 'degC'
    ds['Tc'].attrs['long_name'] = 'average_corrected_RASS_temperature'
    ds['Tc'].attrs['units'] = 'degC'

    # Vertical motion (w)
    ds['W'].attrs['long_name'] = 'average_vertical_wind'
    ds['W'].attrs['units'] = 'm/s'

    # Add time to our dataset
    ds['time'] = time

    # Add in our additional attributes
    ds.attrs['site_identifier'] = site
    ds.attrs['latitude'] = latitude
    ds.attrs['longitude'] = longitude
    ds.attrs['elevation'] = elevation
    ds.attrs['beam_azimuth'] = beam_azimuth
    ds.attrs['revision_number'] = version
    ds.attrs[
        'data_description'
    ] = 'https://psl.noaa.gov/data/obs/data/view_data_type_info.php?SiteID=ctd&DataOperationalID=5855&OperationalID=2371'
    ds.attrs['consensus_average_time'] = consensus_average_time
    ds.attrs['number_of_beams'] = int(number_of_beams)
    ds.attrs['number_of_gates'] = int(number_of_gates)
    ds.attrs['number_of_range_gates'] = int(number_of_range_gates)
    ds.attrs['number_spectral_averages'] = int(number_spectral_averages)
    ds.attrs['pulse_width'] = pulse_width
    ds.attrs['inner_pulse_period'] = inner_pulse_period
    ds.attrs['full_scale_doppler_value'] = full_scale_doppler
    ds.attrs['spacing_of_gates'] = spacing_of_gates

    return ds


def filter_list(list_of_strings):
    """
    Parses a list of strings, remove empty strings, and return a numpy array
    """
    return np.array(list(filter(None, list_of_strings)))


def parse_date_line(list_of_strings):
    """
    Parses the date line in PSL files
    """
    year, month, day, hour, minute, second, utc_offset = filter_list(
        list_of_strings.split(' ')
    ).astype(int)
    year += 2000
    return datetime(year, month, day, hour, minute, second)


def read_psl_surface_met(filenames, conf_file=None):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    NOAA PSL SurfaceMet file.

    Parameters
    ----------
    filenames : str, list of str
        Name of file(s) to read.
    conf_file : str or Pathlib.path
        Default to ./conf/noaapsl_SurfaceMet.yaml
        Filename containing relative or full path to configuration
        YAML file used to describe the file format for each PSL site.
        If the site is not defined in the default file, the default file
        can be coppied to a local location, and the missing site added.
        Then point to that updated configuration file. An issue can be
        opened on GitHub to request the missing site to the configuration
        file.

    Return
    ------
    ds :  Xarray.dataset
        Standard Xarray dataset with the data

    """

    if isinstance(filenames, str):
        site = ospath.basename(filenames)[:3]
    else:
        site = ospath.basename(filenames[0])[:3]

    if conf_file is None:
        conf_file = ospath.join(ospath.dirname(__file__), 'conf', 'noaapsl_SurfaceMet.yaml')

    # Read configuration YAML file
    with open(conf_file, "r") as fp:
        try:
            result = yaml.load(fp, Loader=yaml.FullLoader)
        except AttributeError:
            result = yaml.load(fp)

    # Extract dictionary of just corresponding site
    try:
        result = result[site]
    except KeyError:
        raise RuntimeError(f"Configuration for site '{site}' currently not available. "
                           "You can manually add the site configuration to a copy of "
                           "noaapsl_SurfaceMet.yaml and set conf_file= name of copied file "
                           "until the site is added.")

    # Extract date and time from filename to use in extracting format from YAML file.
    search_result = re.match(r"[a-z]{3}(\d{2})(\d{3})\.(\d{2})m", ospath.basename(filenames[0]))
    yy, doy, hh = search_result.groups()
    if yy > '70':
        yy = f"19{yy}"
    else:
        yy = f"20{yy}"

    # Extract location information from configuration file.
    try:
        location_info = result['info']
    except KeyError:
        location_info = None

    # Loop through each date range for the site to extract the correct file format from conf file.
    file_datetime = datetime.strptime(f'{yy}-01-01', '%Y-%m-%d') + timedelta(int(doy) - 1) + timedelta(hours=int(hh))
    for ii in result.keys():
        if ii == 'info':
            continue

        date_range = [datetime.strptime(jj, '%Y-%m-%d %H:%M:%S') for jj in result[ii]['_date_range']]
        if file_datetime >= date_range[0] and file_datetime <= date_range[1]:
            result = result[ii]
            del result['_date_range']
            break

    # Read data files by passing in column names from configuration file.
    data = read_csv(filenames, column_names=list(result.keys()))

    # Calculate numpy datetime64 values from first 4 columns of the data file.
    time = np.array(data['Year'].values - 1970, dtype='datetime64[Y]')
    day = np.array(np.array(data['J_day'].values - 1 , dtype='timedelta64[D]'))
    hourmin = data['HoursMinutes'].values + 10000
    hour = [int(str(ii)[1:3]) for ii in hourmin]
    hour = np.array(hour, dtype='timedelta64[h]')
    minute = [int(str(ii)[3:]) for ii in hourmin]
    minute = np.array(minute, dtype='timedelta64[m]')
    time = time + day + hour + minute

    # Update Dataset to use "time" coordinate and assigned calculated times
    data = data.assign_coords(index=time)
    data = data.rename(index='time')

    # Loop through configuraton dictionary and apply attributes or
    # perform action for specific attributes.
    for var_name in result:
        for key, value in result[var_name].items():
            if key == '_delete' and value is True:
                del data[var_name]
                continue

            if key == '_type':
                dtype = result[var_name][key]
                data[var_name] = data[var_name].astype(dtype)
                continue

            if key == '_missing_value':
                data_values = data[var_name].values
                data_values[data_values == result[var_name][key]] = np.nan
                data[var_name].values = data_values
                continue

            data[var_name].attrs[key] = value

    # Add location information to Dataset
    if location_info is not None:
        data.attrs['location_description'] = location_info['name']
        for var_name in ['lat', 'lon', 'alt']:
            value = location_info[var_name]['value']
            del location_info[var_name]['value']
            data[var_name] = xr.DataArray(data=value, attrs=location_info[var_name])

    return data
