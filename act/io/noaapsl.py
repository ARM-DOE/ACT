"""
Modules for reading in NOAA PSL data.
"""

from datetime import datetime
from itertools import groupby

import fsspec
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

    # The first entry should be the station identifier (ex. CTD)
    potential_site = df[0][0]

    # Get location of where each table begins
    index_list = df[0] == potential_site
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
    filename : str
        Name of file(s) to read.

    Return
    ------
    ds :  Xarray.dataset
        Standard Xarray dataset with the data

    """

    # Open the file, read in the lines as a list, and return that list
    file = fsspec.open(filepath).open()
    lines = file.readlines()
    lines = [x.decode().rstrip()[:] for x in lines][1:]

    # Separate sections based on the $ separator in the file
    sections_of_file = (list(g) for _, g in groupby(lines, key='$'.__ne__))

    # Count how many lines need to be skipped when reading into pandas
    start_line = 0
    list_of_datasets = []
    for section in sections_of_file:
        if section[0] != '$':
            list_of_datasets.append(
                _parse_psl_temperature_lines(filepath, section, line_offset=start_line)
            )
        start_line += len(section)

    # Merge the resultant datasets together
    return xr.concat(list_of_datasets, dim='time').transpose('HT', 'time')


def _parse_psl_temperature_lines(filepath, lines, line_offset=0):
    """
    Reads lines related to temperature in a psl file

    Parameters
    ----------
    filename : str
        Name of file(s) to read.

    lines = list
      List of strings containing the lines to parse

    line_offset = int (default = 0)
      Offset to start reading the pandas data table

    Returns
    -------
    ds = xr.Dataset
      Xarray dataset with temperature data

    """
    # 1 - site
    site = lines[0]

    # 2 - datetype
    datatype, _, version = filter_list(lines[1].split(' '))

    # 3 - station lat, lon, elevation
    latitude, longitude, elevation = filter_list(lines[2].split('  ')).astype(float)

    # 4 - year, month, day, hour, minute, second, utc
    time = parse_date_line(lines[3])

    # 5 - Consensus averaging time, number of beams, number of range gates
    consensus_average_time, number_of_beams, number_of_range_gates = filter_list(
        lines[4].split('  ')
    ).astype(int)

    # 7 - number of coherent integrations, number of spectral averages, pulse width, indder pulse period
    (
        number_coherent_integrations,
        number_spectral_averages,
        pulse_width,
        inner_pulse_period,
    ) = filter_list(lines[6].split(' ')).astype(int)

    # 8 - full-scale doppler value, delay to first gate, number of gates, spacing of gates
    full_scale_doppler, delay_first_gate, number_of_gates, spacing_of_gates = filter_list(
        lines[7].split(' ')
    ).astype(float)

    # 9 - beam azimuth (degrees clockwise from north)
    beam_azimuth, beam_elevation = filter_list(lines[8].split(' ')).astype(float)

    # Read in the data table section using pandas
    df = pd.read_csv(filepath, skiprows=line_offset + 10, delim_whitespace=True)

    # Only read in the number of rows for a given set of gates
    df = df.iloc[: int(number_of_gates)]

    # Grab a list of valid columns, exept time
    columns = set(list(df.columns)) - {'time'}

    # Set the data types to be floats
    df = df[list(columns)].astype(float)

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


def read_psl_parsivel(files):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    NOAA PSL parsivel

    Parameters
    ----------
    files : str or list
        Name of file(s) or urls to read.

    Return
    ------
    obj : Xarray.dataset
        Standard Xarray dataset with the data for the parsivel

    """

    # Define the names for the variables
    names = ['time', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12',
             'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24',
             'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'blackout', 'good', 'bad',
             'number_detected_particles', 'precip_rate', 'precip_amount', 'precip_accumulation',
             'equivalent_radar_reflectivity', 'number_in_error', 'dirty', 'very_dirty', 'damaged',
             'laserband_amplitude', 'laserband_amplitude_stdev', 'sensor_temperature', 'sensor_temperature_stdev',
             'sensor_voltage', 'sensor_voltage_stdev', 'heating_current', 'heating_current_stdev', 'number_rain_particles',
             'number_non_rain_particles', 'number_ambiguous_particles', 'precip_type']

    # Define the particle sizes and class width sizes based on
    # https://psl.noaa.gov/data/obs/data/view_data_type_info.php?SiteID=ctd&DataOperationalID=5890
    vol_equiv_diam = [0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375,
                      1.625, 1.875, 2.125, 2.375, 2.75, 3.25, 3.75, 4.25, 4.75, 5.5, 6.5, 7.5, 8.5,
                      9.5, 11.0, 13.0, 15.0, 17.0, 19.0, 21.5, 24.5]
    class_size_width = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                        0.250, 0.250, 0.250, 0.250, 0.250, 0.5, 0.5, 0.5, 0.5, 0.5,
                        1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0]

    if not isinstance(files, list):
        files = [files]

    # Loop through each file or url and append the dataframe into data for concatenations
    data = []
    end_time = []
    for f in files:
        df = pd.read_table(f, skiprows=[0, 1, 2], names=names, index_col=0, sep='\s+')
        # Reading the table twice to get the date so it can be parsed appropriately
        date = pd.read_table(f, nrows=0).to_string().split(' ')[-3]
        time = df.index
        start_time = []
        form = '%y%j%H:%M:%S:%f'
        for t in time:
            start_time.append(pd.to_datetime(date + ':' + t.split('-')[0], format=form))
            end_time.append(pd.to_datetime(date + ':' + t.split('-')[1], format=form))
        df.index = start_time
        data.append(df)

    df = pd.concat(data)

    # Create a 2D size distribution variable from all the B* variables
    dsd = []
    for n in names:
        if 'B' not in n:
            continue
        dsd.append(list(df[n]))

    # Convert the dataframe to xarray DataSet and add variables
    obj = df.to_xarray()
    obj = obj.rename({'index': 'time'})
    long_name = 'Drop Size Distribution'
    attrs = {'long_name': long_name, 'units': 'count'}
    da = xr.DataArray(np.transpose(dsd), dims=['time', 'particle_size'], coords=[obj['time'].values, vol_equiv_diam])
    obj['number_density_drops'] = da

    attrs = {'long_name': 'Particle class size average', 'units': 'mm'}
    da = xr.DataArray(class_size_width, dims=['particle_size'], coords=[vol_equiv_diam], attrs=attrs)
    obj['class_size_width'] = da

    attrs = {'long_name': 'Class size width', 'units': 'mm'}
    da = xr.DataArray(vol_equiv_diam, dims=['particle_size'], coords=[vol_equiv_diam], attrs=attrs)
    obj['particle_size'] = da

    attrs = {'long_name': 'End time of averaging interval'}
    da = xr.DataArray(end_time, dims=['time'], coords=[obj['time'].values], attrs=attrs)
    obj['interval_end_time'] = da

    # Define the attribuets and metadata and add into the DataSet
    attrs = {'blackout': {'long_name': 'Number of samples excluded during PC clock sync', 'units': 'count'},
             'good': {'long_name': 'Number of samples that passed QC checks', 'units': 'count'},
             'bad': {'long_name': 'Number of samples that failed QC checks', 'units': 'count'},
             'number_detected_particles': {'long_name': 'Total number of detected particles', 'units': 'count'},
             'precip_rate': {'long_name': 'Precipitation rate', 'units': 'mm/hr'},
             'precip_amount': {'long_name': 'Interval accumulation', 'units': 'mm'},
             'precip_accumulation': {'long_name': 'Event accumulation', 'units': 'mm'},
             'equivalent_radar_reflectivity': {'long_name': 'Radar Reflectivity', 'units': 'dB'},
             'number_in_error': {'long_name': 'Number of samples that were reported dirt, very dirty, or damaged', 'units': 'count'},
             'dirty': {'long_name': 'Laser glass is dirty but measurement is still possible', 'units': 'unitless'},
             'very_dirty': {'long_name': 'Laser glass is dirty, partially covered no further measurements are possible', 'units': 'unitless'},
             'damaged': {'long_name': 'Laser damaged', 'units': 'unitless'},
             'laserband_amplitude': {'long_name': 'Average signal amplitude of the laser strip', 'units': 'unitless'},
             'laserband_amplitude_stdev': {'long_name': 'Standard deviation of the signal amplitude of the laser strip', 'units': 'unitless'},
             'sensor_temperature': {'long_name': 'Average sensor temperature', 'units': 'degC'},
             'sensor_temperature_stdev': {'long_name': 'Standard deviation of sensor temperature', 'units': 'degC'},
             'sensor_voltage': {'long_name': 'Sensor power supply voltage', 'units': 'V'},
             'sensor_voltage_stdev': {'long_name': 'Standard deviation of the sensor power supply voltage', 'units': 'V'},
             'heating_current': {'long_name': 'Average heating system current', 'units': 'A'},
             'heating_current_stdev': {'long_name': 'Standard deviation of heating system current', 'units': 'A'},
             'number_rain_particles': {'long_name': 'Number of particles detected as rain', 'units': 'unitless'},
             'number_non_rain_particles': {'long_name': 'Number of particles detected not as rain', 'units': 'unitless'},
             'number_ambiguous_particles': {'long_name': 'Number of particles detected as ambiguous', 'units': 'unitless'},
             'precip_type': {'long_name': 'Precipitation type (1=rain; 2=mixed; 3=snow)', 'units': 'unitless'},
             'number_density_drops': {'long_name': 'Drop Size Distribution', 'units': 'count'}}

    for v in obj:
        if v in attrs:
            obj[v].attrs = attrs[v]

    return obj
