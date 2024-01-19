"""
Modules for reading in NOAA PSL data.
"""

import datetime as dt
import re
from datetime import datetime, timedelta
from itertools import groupby
from os import path as ospath

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .text import read_csv


def read_psl_wind_profiler(filepath, transpose=True):
    """
    Returns two `xarray.Datasets` with stored data and metadata from a
    user-defined NOAA PSL wind profiler file each containing
    a different mode. This works for both 449 MHz and 915 MHz Weber
    Wuertz and Weber Wuertz sub-hourly files.

    Parameters
    ----------
    filepath : str
        Name of file(s) to read.
    transpose : bool
        True to transpose the data.

    Return
    ------
    mode_one_ds : xarray.Dataset
        Standard Xarray dataset with the first mode data.
    mode_two_ds : xarray.Dataset
        Standard Xarray dataset with the second mode data.

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
                _parse_psl_wind_lines(filepath, section, line_offset=start_line)
            )
        start_line += len(section)

    # Return two datasets for each mode and the merge of datasets of the
    # same mode.
    mode_one_ds = xr.concat(list_of_datasets[0::2], dim='time')
    mode_two_ds = xr.concat(list_of_datasets[1::2], dim='time')
    if transpose:
        mode_one_ds = mode_one_ds.transpose('HT', 'time')
        mode_two_ds = mode_two_ds.transpose('HT', 'time')
    return mode_one_ds, mode_two_ds


def read_psl_wind_profiler_temperature(filepath, transpose=True):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    NOAA PSL wind profiler temperature file.

    Parameters
    ----------
    filepath : str
        Name of file(s) to read.
    transpose : bool
        True to transpose the data.

    Return
    ------
    ds : xarray.Dataset
        Standard Xarray dataset with the data.

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
    if transpose:
        return xr.concat(list_of_datasets, dim='time').transpose('HT', 'time')
    else:
        return xr.concat(list_of_datasets, dim='time')


def _parse_psl_wind_lines(filepath, lines, line_offset=0):
    """
    Reads lines related to wind in a psl file.

    Parameters
    ----------
    filepath : str
        Name of file(s) to read.
    lines : list
      List of strings containing the lines to parse.
    line_offset : int (default = 0)
      Offset to start reading the pandas data table.

    Returns
    -------
    ds : xarray.Dataset
      Xarray dataset with wind data.

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

    # 7 - number of coherent integrations, number of spectral averages,
    # pulse width, inner pulse period'
    # Values duplicate as oblique and vertical values
    (
        number_coherent_integrations_obl,
        number_coherent_integrations_vert,
        number_spectral_averages_obl,
        number_spectral_averages_vert,
        pulse_width_obl,
        pulse_width_vert,
        inner_pulse_period_obl,
        inner_pulse_period_vert,
    ) = filter_list(lines[6].split(' ')).astype(int)

    # 8 - full-scale doppler value, delay to first gate, number of gates,
    # spacing of gates. Values duplicate as oblique and vertical values.
    (
        full_scale_doppler_obl,
        full_scale_doppler_vert,
        beam_vertical_correction,
        delay_first_gate_obl,
        delay_first_gate_vert,
        number_of_gates_obl,
        number_of_gates_vert,
        spacing_of_gates_obl,
        spacing_of_gates_vert,
    ) = filter_list(lines[7].split(' ')).astype(float)

    # 9 - beam azimuth (degrees clockwise from north)
    (
        beam_azimuth1,
        beam_elevation1,
        beam_azimuth2,
        beam_elevation2,
        beam_azimuth3,
        beam_elevation3,
    ) = filter_list(lines[8].split(' ')).astype(float)

    beam_azimuth = np.array([beam_azimuth1, beam_azimuth2, beam_azimuth3], dtype='float32')
    beam_elevation = np.array([beam_elevation1, beam_elevation2, beam_elevation3], dtype='float32')

    # Read in the data table section using pandas
    df = pd.read_csv(filepath, skiprows=line_offset + 10, delim_whitespace=True)

    # Only read in the number of rows for a given set of gates
    df = df.iloc[: int(number_of_range_gates)]

    # Grab a list of valid columns, except time
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
            'RAD': 'RAD1',
            'RAD.1': 'RAD2',
            'RAD.2': 'RAD3',
            'CNT': 'CNT1',
            'CNT.1': 'CNT2',
            'CNT.2': 'CNT3',
            'SNR': 'SNR1',
            'SNR.1': 'SNR2',
            'SNR.2': 'SNR3',
            'QC': 'QC1',
            'QC.1': 'QC2',
            'QC.2': 'QC3',
        }
    )

    # Convert to an xaray dataset
    ds = df.to_xarray()

    # Add attributes to variables
    # Height
    ds['HT'].attrs['long_name'] = 'height_above_ground'
    ds['HT'].attrs['units'] = 'km'

    # Add time to our dataset
    ds['time'] = time

    # Add in our additional attributes
    ds.attrs['site_identifier'] = site.strip()
    ds.attrs['data_type'] = datatype
    ds.attrs['latitude'] = latitude
    ds.attrs['longitude'] = longitude
    ds.attrs['elevation'] = elevation
    ds.attrs['beam_elevation'] = beam_elevation
    ds.attrs['beam_azimuth'] = beam_azimuth
    ds.attrs['revision_number'] = version
    ds.attrs[
        'data_description'
    ] = 'https://psl.noaa.gov/data/obs/data/view_data_type_info.php?SiteID=ctd&DataOperationalID=5855&OperationalID=2371'
    ds.attrs['consensus_average_time'] = consensus_average_time
    ds.attrs['oblique-beam_vertical_correction'] = int(beam_vertical_correction)
    ds.attrs['number_of_beams'] = int(number_of_beams)
    ds.attrs['number_of_range_gates'] = int(number_of_range_gates)

    # Handle oblique and vertical attributes.
    ds.attrs['number_of_gates_oblique'] = int(number_of_gates_obl)
    ds.attrs['number_of_gates_vertical'] = int(number_of_gates_vert)
    ds.attrs['number_spectral_averages_oblique'] = int(number_spectral_averages_obl)
    ds.attrs['number_spectral_averages_vertical'] = int(number_spectral_averages_vert)
    ds.attrs['pulse_width_oblique'] = int(pulse_width_obl)
    ds.attrs['pulse_width_vertical'] = int(pulse_width_vert)
    ds.attrs['inner_pulse_period_oblique'] = int(inner_pulse_period_obl)
    ds.attrs['inner_pulse_period_vertical'] = int(inner_pulse_period_vert)
    ds.attrs['full_scale_doppler_value_oblique'] = float(full_scale_doppler_obl)
    ds.attrs['full_scale_doppler_value_vertical'] = float(full_scale_doppler_vert)
    ds.attrs['delay_to_first_gate_oblique'] = int(delay_first_gate_obl)
    ds.attrs['delay_to_first_gate_vertical'] = int(delay_first_gate_vert)
    ds.attrs['spacing_of_gates_oblique'] = int(spacing_of_gates_obl)
    ds.attrs['spacing_of_gates_vertical'] = int(spacing_of_gates_vert)
    return ds


def _parse_psl_temperature_lines(filepath, lines, line_offset=0):
    """
    Reads lines related to temperature in a psl file.

    Parameters
    ----------
    filepath : str
        Name of file(s) to read.
    lines : list
      List of strings containing the lines to parse.
    line_offset : int (default = 0)
      Offset to start reading the pandas data table.

    Returns
    -------
    ds : xarray.Dataset
      Xarray dataset with temperature data.

    """
    # 1 - site
    site = lines[0]

    # 2 - datetype
    datatype, _, version = filter_list(lines[1].split(' '))

    # 3 - station lat, lon, elevation
    latitude, longitude, elevation = filter_list(lines[2].split('  ')).astype(float)

    # 4 - year, month, day, hour, minute, second, utc
    time = parse_date_line(lines[3])

    # 5 - Consensus averaging time, number of beams, number of range gates.
    consensus_average_time, number_of_beams, number_of_range_gates = filter_list(
        lines[4].split('  ')
    ).astype(int)

    # 7 - number of coherent integrations, number of spectral averages,
    # pulse width, inner pulse period.
    (
        number_coherent_integrations,
        number_spectral_averages,
        pulse_width,
        inner_pulse_period,
    ) = filter_list(lines[6].split(' ')).astype(int)

    # 8 - full-scale doppler value, delay to first gate, number of gates,
    # spacing of gates.
    full_scale_doppler, delay_first_gate, number_of_gates, spacing_of_gates = filter_list(
        lines[7].split(' ')
    ).astype(float)

    # 9 - beam azimuth (degrees clockwise from north)
    beam_azimuth, beam_elevation = filter_list(lines[8].split(' ')).astype(float)

    # Read in the data table section using pandas
    df = pd.read_csv(filepath, skiprows=line_offset + 10, delim_whitespace=True)

    # Only read in the number of rows for a given set of gates
    df = df.iloc[: int(number_of_gates)]

    # Grab a list of valid columns, except time
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
    ds.attrs['site_identifier'] = site.strip()
    ds.attrs['data_type'] = datatype
    ds.attrs['latitude'] = latitude
    ds.attrs['longitude'] = longitude
    ds.attrs['elevation'] = elevation
    ds.attrs['beam_elevation'] = beam_elevation
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
        can be copied to a local location, and the missing site added.
        Then point to that updated configuration file. An issue can be
        opened on GitHub to request the missing site to the configuration
        file.

    Return
    ------
    ds :  xarray.Dataset
        Standard Xarray dataset with the data

    """

    if isinstance(filenames, str):
        site = ospath.basename(filenames)[:3]
    else:
        site = ospath.basename(filenames[0])[:3]

    if conf_file is None:
        conf_file = ospath.join(ospath.dirname(__file__), 'conf', 'noaapsl_SurfaceMet.yaml')

    # Read configuration YAML file
    with open(conf_file) as fp:
        try:
            result = yaml.load(fp, Loader=yaml.FullLoader)
        except AttributeError:
            result = yaml.load(fp)

    # Extract dictionary of just corresponding site
    try:
        result = result[site]
    except KeyError:
        raise RuntimeError(
            f"Configuration for site '{site}' currently not available. "
            'You can manually add the site configuration to a copy of '
            'noaapsl_SurfaceMet.yaml and set conf_file= name of copied file '
            'until the site is added.'
        )

    # Extract date and time from filename to use in extracting format from YAML file.
    search_result = re.match(r'[a-z]{3}(\d{2})(\d{3})\.(\d{2})m', ospath.basename(filenames[0]))
    yy, doy, hh = search_result.groups()
    if yy > '70':
        yy = f'19{yy}'
    else:
        yy = f'20{yy}'

    # Extract location information from configuration file.
    try:
        location_info = result['info']
    except KeyError:
        location_info = None

    # Loop through each date range for the site to extract the correct file format from conf file.
    file_datetime = (
        datetime.strptime(f'{yy}-01-01', '%Y-%m-%d')
        + timedelta(int(doy) - 1)
        + timedelta(hours=int(hh))
    )
    for ii in result.keys():
        if ii == 'info':
            continue

        date_range = [
            datetime.strptime(jj, '%Y-%m-%d %H:%M:%S') for jj in result[ii]['_date_range']
        ]
        if file_datetime >= date_range[0] and file_datetime <= date_range[1]:
            result = result[ii]
            del result['_date_range']
            break

    # Read data files by passing in column names from configuration file.
    ds = read_csv(filenames, column_names=list(result.keys()))

    # Calculate numpy datetime64 values from first 4 columns of the data file.
    time = np.array(ds['Year'].values - 1970, dtype='datetime64[Y]')
    day = np.array(np.array(ds['J_day'].values - 1, dtype='timedelta64[D]'))
    hourmin = ds['HoursMinutes'].values + 10000
    hour = [int(str(ii)[1:3]) for ii in hourmin]
    hour = np.array(hour, dtype='timedelta64[h]')
    minute = [int(str(ii)[3:]) for ii in hourmin]
    minute = np.array(minute, dtype='timedelta64[m]')
    time = time + day + hour + minute
    time = time.astype('datetime64[ns]')
    # Update Dataset to use "time" coordinate and assigned calculated times
    ds = ds.assign_coords(index=time)
    ds = ds.rename(index='time')

    # Loop through configuraton dictionary and apply attributes or
    # perform action for specific attributes.
    for var_name in result:
        for key, value in result[var_name].items():
            if key == '_delete' and value is True:
                del ds[var_name]
                continue

            if key == '_type':
                dtype = result[var_name][key]
                ds[var_name] = ds[var_name].astype(dtype)
                continue

            if key == '_missing_value':
                data_values = ds[var_name].values
                data_values[data_values == result[var_name][key]] = np.nan
                ds[var_name].values = data_values
                continue

            ds[var_name].attrs[key] = value

    # Add location information to Dataset
    if location_info is not None:
        ds.attrs['location_description'] = location_info['name']
        for var_name in ['lat', 'lon', 'alt']:
            value = location_info[var_name]['value']
            del location_info[var_name]['value']
            ds[var_name] = xr.DataArray(data=value, attrs=location_info[var_name])

    return ds


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
    ds : xarray.Dataset
        Standard Xarray dataset with the data for the parsivel

    """

    # Define the names for the variables
    names = [
        'time',
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B9',
        'B10',
        'B11',
        'B12',
        'B13',
        'B14',
        'B15',
        'B16',
        'B17',
        'B18',
        'B19',
        'B20',
        'B21',
        'B22',
        'B23',
        'B24',
        'B25',
        'B26',
        'B27',
        'B28',
        'B29',
        'B30',
        'B31',
        'B32',
        'blackout',
        'good',
        'bad',
        'number_detected_particles',
        'precip_rate',
        'precip_amount',
        'precip_accumulation',
        'equivalent_radar_reflectivity',
        'number_in_error',
        'dirty',
        'very_dirty',
        'damaged',
        'laserband_amplitude',
        'laserband_amplitude_stdev',
        'sensor_temperature',
        'sensor_temperature_stdev',
        'sensor_voltage',
        'sensor_voltage_stdev',
        'heating_current',
        'heating_current_stdev',
        'number_rain_particles',
        'number_non_rain_particles',
        'number_ambiguous_particles',
        'precip_type',
    ]

    # Define the particle sizes and class width sizes based on
    # https://psl.noaa.gov/data/obs/data/view_data_type_info.php?SiteID=ctd&DataOperationalID=5890
    vol_equiv_diam = [
        0.062,
        0.187,
        0.312,
        0.437,
        0.562,
        0.687,
        0.812,
        0.937,
        1.062,
        1.187,
        1.375,
        1.625,
        1.875,
        2.125,
        2.375,
        2.75,
        3.25,
        3.75,
        4.25,
        4.75,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
        11.0,
        13.0,
        15.0,
        17.0,
        19.0,
        21.5,
        24.5,
    ]
    class_size_width = [
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.125,
        0.250,
        0.250,
        0.250,
        0.250,
        0.250,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
    ]

    if not isinstance(files, list):
        files = [files]

    # Loop through each file or url and append the dataframe into data for concatenations
    data = []
    end_time = []
    for f in files:
        df = pd.read_table(f, skiprows=[0, 1, 2], names=names, index_col=0, sep=r'\s+')
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
    ds = df.to_xarray()
    ds = ds.rename({'index': 'time'})
    long_name = 'Drop Size Distribution'
    attrs = {'long_name': long_name, 'units': 'count'}
    da = xr.DataArray(
        np.transpose(dsd),
        dims=['time', 'particle_size'],
        coords=[ds['time'].values, vol_equiv_diam],
    )
    ds['number_density_drops'] = da

    attrs = {'long_name': 'Particle class size average', 'units': 'mm'}
    da = xr.DataArray(
        class_size_width, dims=['particle_size'], coords=[vol_equiv_diam], attrs=attrs
    )
    ds['class_size_width'] = da

    attrs = {'long_name': 'Class size width', 'units': 'mm'}
    da = xr.DataArray(vol_equiv_diam, dims=['particle_size'], coords=[vol_equiv_diam], attrs=attrs)
    ds['particle_size'] = da

    attrs = {'long_name': 'End time of averaging interval'}
    da = xr.DataArray(end_time, dims=['time'], coords=[ds['time'].values], attrs=attrs)
    ds['interval_end_time'] = da

    # Define the attribuets and metadata and add into the DataSet
    attrs = {
        'blackout': {
            'long_name': 'Number of samples excluded during PC clock sync',
            'units': 'count',
        },
        'good': {'long_name': 'Number of samples that passed QC checks', 'units': 'count'},
        'bad': {'long_name': 'Number of samples that failed QC checks', 'units': 'count'},
        'number_detected_particles': {
            'long_name': 'Total number of detected particles',
            'units': 'count',
        },
        'precip_rate': {'long_name': 'Precipitation rate', 'units': 'mm/hr'},
        'precip_amount': {'long_name': 'Interval accumulation', 'units': 'mm'},
        'precip_accumulation': {'long_name': 'Event accumulation', 'units': 'mm'},
        'equivalent_radar_reflectivity': {'long_name': 'Radar Reflectivity', 'units': 'dB'},
        'number_in_error': {
            'long_name': 'Number of samples that were reported dirt, very dirty, or damaged',
            'units': 'count',
        },
        'dirty': {
            'long_name': 'Laser glass is dirty but measurement is still possible',
            'units': 'unitless',
        },
        'very_dirty': {
            'long_name': 'Laser glass is dirty, partially covered no further measurements are possible',
            'units': 'unitless',
        },
        'damaged': {'long_name': 'Laser damaged', 'units': 'unitless'},
        'laserband_amplitude': {
            'long_name': 'Average signal amplitude of the laser strip',
            'units': 'unitless',
        },
        'laserband_amplitude_stdev': {
            'long_name': 'Standard deviation of the signal amplitude of the laser strip',
            'units': 'unitless',
        },
        'sensor_temperature': {'long_name': 'Average sensor temperature', 'units': 'degC'},
        'sensor_temperature_stdev': {
            'long_name': 'Standard deviation of sensor temperature',
            'units': 'degC',
        },
        'sensor_voltage': {'long_name': 'Sensor power supply voltage', 'units': 'V'},
        'sensor_voltage_stdev': {
            'long_name': 'Standard deviation of the sensor power supply voltage',
            'units': 'V',
        },
        'heating_current': {'long_name': 'Average heating system current', 'units': 'A'},
        'heating_current_stdev': {
            'long_name': 'Standard deviation of heating system current',
            'units': 'A',
        },
        'number_rain_particles': {
            'long_name': 'Number of particles detected as rain',
            'units': 'unitless',
        },
        'number_non_rain_particles': {
            'long_name': 'Number of particles detected not as rain',
            'units': 'unitless',
        },
        'number_ambiguous_particles': {
            'long_name': 'Number of particles detected as ambiguous',
            'units': 'unitless',
        },
        'precip_type': {
            'long_name': 'Precipitation type (1=rain; 2=mixed; 3=snow)',
            'units': 'unitless',
        },
        'number_density_drops': {'long_name': 'Drop Size Distribution', 'units': 'count'},
    }

    for v in ds:
        if v in attrs:
            ds[v].attrs = attrs[v]

    return ds


def read_psl_radar_fmcw_moment(files):
    """
    Returns `xarray.Dataset` with stored data and metadata from
    NOAA PSL FMCW Radar files. See References section for details.

    Parameters
    ----------
    files : str or list
        Name of file(s) to read.  Currently does not support reading URLs but files can
        be downloaded easily using the act.discovery.download_noaa_psl_data function.

    Return
    ------
    ds : xarray.Dataset
        Standard Xarray dataset with the data for the parsivel

    References
    ----------
    Johnston, Paul E., James R. Jordan, Allen B. White, David A. Carter, David M. Costa, and Thomas E. Ayers.
        "The NOAA FM-CW snow-level radar." Journal of Atmospheric and Oceanic Technology 34, no. 2 (2017): 249-267.

    """

    ds = _parse_psl_radar_moments(files)

    return ds


def read_psl_radar_sband_moment(files):
    """
    Returns `xarray.Dataset` with stored data and metadata from
    NOAA PSL S-band Radar files.

    Parameters
    ----------
    files : str or list
        Name of file(s) to read.  Currently does not support reading URLs but files can
        be downloaded easily using the act.discovery.download_noaa_psl_data function.

    Return
    ------
    ds : xarray.Dataset
        Standard Xarray dataset with the data for the parsivel

    """

    ds = _parse_psl_radar_moments(files)

    return ds


def _parse_psl_radar_moments(files):
    """
    Returns `xarray.Dataset` with stored data and metadata from
    NOAA PSL FMCW and S-Band Radar files.

    Parameters
    ----------
    files : str or list
        Name of file(s) to read.  Currently does not support reading URLs but files can
        be downloaded easily using the act.discovery.download_noaa_psl_data function.

    Return
    ------
    ds : xarray.Dataset
        Standard Xarray dataset with the data for the parsivel

    """
    # Set the initial dictionary to convert to xarray dataset
    data = {
        'site': {'dims': ['file'], 'data': [], 'attrs': {'long_name': 'NOAA site code'}},
        'lat': {
            'dims': ['file'],
            'data': [],
            'attrs': {'long_name': 'North Latitude', 'units': 'degree_N'},
        },
        'lon': {
            'dims': ['file'],
            'data': [],
            'attrs': {'long_name': 'East Longitude', 'units': 'degree_E'},
        },
        'alt': {
            'dims': ['file'],
            'data': [],
            'attrs': {'long_name': 'Altitude above mean sea level', 'units': 'm'},
        },
        'freq': {
            'dims': ['file'],
            'data': [],
            'attrs': {'long_name': 'Operating Frequency; Ignore for FMCW', 'units': 'Hz'},
        },
        'azimuth': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Azimuth angle', 'units': 'deg'},
        },
        'elevation': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Elevation angle', 'units': 'deg'},
        },
        'beam_direction_code': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Beam direction code', 'units': ''},
        },
        'year': {'dims': ['time'], 'data': [], 'attrs': {'long_name': '2-digit year', 'units': ''}},
        'day_of_year': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Day of the year', 'units': ''},
        },
        'hour': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Hour of the day', 'units': ''},
        },
        'minute': {'dims': ['time'], 'data': [], 'attrs': {'long_name': 'Minutes', 'units': ''}},
        'second': {'dims': ['time'], 'data': [], 'attrs': {'long_name': 'Seconds', 'units': ''}},
        'interpulse_period': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Interpulse Period', 'units': 'ms'},
        },
        'pulse_width': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Pulse width', 'units': 'ns'},
        },
        'first_range_gate': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Range to first range gate', 'units': 'm'},
        },
        'range_between_gates': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Distance between range gates', 'units': 'm'},
        },
        'n_gates': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Number of range gates', 'units': 'count'},
        },
        'n_coherent_integration': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Number of cohrent integration', 'units': 'count'},
        },
        'n_averaged_spectra': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Number of average spectra', 'units': 'count'},
        },
        'n_points_spectrum': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Number of points in spectra', 'units': 'count'},
        },
        'n_code_bits': {
            'dims': ['time'],
            'data': [],
            'attrs': {'long_name': 'Number of code bits', 'units': 'count'},
        },
        'radial_velocity': {
            'dims': ['time', 'range'],
            'data': [],
            'attrs': {'long_name': 'Radial velocity', 'units': 'm/s'},
        },
        'snr': {
            'dims': ['time', 'range'],
            'data': [],
            'attrs': {'long_name': 'Signal-to-noise ratio - not range corrected', 'units': 'dB'},
        },
        'signal_power': {
            'dims': ['time', 'range'],
            'data': [],
            'attrs': {'long_name': 'Signal Power - not range corrected', 'units': 'dB'},
        },
        'spectral_width': {
            'dims': ['time', 'range'],
            'data': [],
            'attrs': {'long_name': 'Spectral width', 'units': 'm/s'},
        },
        'noise_amplitude': {
            'dims': ['time', 'range'],
            'data': [],
            'attrs': {'long_name': 'noise_amplitude', 'units': 'dB'},
        },
        'qc_variable': {
            'dims': ['time', 'range'],
            'data': [],
            'attrs': {'long_name': 'QC Value - not used', 'units': ''},
        },
        'time': {'dims': ['time'], 'data': [], 'attrs': {'long_name': 'Datetime', 'units': ''}},
        'range': {'dims': ['range'], 'data': [], 'attrs': {'long_name': 'Range', 'units': 'm'}},
        'reflectivity_uncalibrated': {
            'dims': ['time', 'range'],
            'data': [],
            'attrs': {'long_name': 'Range', 'units': 'dB'},
        },
    }

    # Separate out the names as they will be accessed in different parts of the code
    h1_names = ['site', 'lat', 'lon', 'alt', 'freq']
    h2_names = [
        'azimuth',
        'elevation',
        'beam_direction_code',
        'year',
        'day_of_year',
        'hour',
        'minute',
        'second',
    ]
    h3_names = [
        'interpulse_period',
        'pulse_width',
        'first_range_gate',
        'range_between_gates',
        'n_gates',
        'n_coherent_integration',
        'n_averaged_spectra',
        'n_points_spectrum',
        'n_code_bits',
    ]
    names = {
        'radial_velocity': 0,
        'snr': 1,
        'signal_power': 2,
        'spectral_width': 3,
        'noise_amplitude': 4,
        'qc_variable': 5,
    }

    # If file is a string, convert to list for handling.
    if not isinstance(files, list):
        files = [files]

    # Run through each file and read the data in
    for f in files:
        # Read in the first line of the file which has site, lat, lon, etc...
        df = str(pd.read_table(f, nrows=0).columns[0]).split(' ')
        ctr = 0
        for d in df:
            if len(d) > 0:
                if d == 'lat' or d == 'lon':
                    data[h1_names[ctr]]['data'].append(float(d) / 100.0)
                else:
                    data[h1_names[ctr]]['data'].append(d)
                ctr += 1

        # Set counts and errors
        error = False
        error_ct = 0
        ct = 0
        # Loop through while there's no errors i.e. eof
        while error is False:
            try:
                # Read in the initial headers to get information used to parse data
                if ct == 0:
                    df = str(pd.read_table(f, nrows=0, skiprows=[0]).columns[0]).split(' ')
                    ctr = 0
                    for d in df:
                        if len(d) > 0:
                            data[h2_names[ctr]]['data'].append(d)
                            ctr += 1
                    # Read in third row of header information
                    df = str(pd.read_table(f, nrows=0, skiprows=[0, 1]).columns[0]).split(' ')
                    ctr = 0
                    for d in df:
                        if len(d) > 0:
                            data[h3_names[ctr]]['data'].append(d)
                            ctr += 1
                    # Read in the data based on number of gates
                    df = pd.read_csv(
                        f,
                        skiprows=[0, 1, 2],
                        nrows=int(data['n_gates']['data'][-1]) - 1,
                        delim_whitespace=True,
                        names=list(names.keys()),
                    )
                    index2 = 0
                else:
                    # Set indices for parsing data, reading 2 headers and then the columns of data
                    index1 = ct * int(data['n_gates']['data'][-1])
                    index2 = index1 + int(data['n_gates']['data'][-1]) + 2 * ct + 4
                    df = str(
                        pd.read_table(f, nrows=0, skiprows=list(range(index2 - 1))).columns[0]
                    ).split(' ')
                    ctr = 0
                    for d in df:
                        if len(d) > 0:
                            data[h2_names[ctr]]['data'].append(d)
                            ctr += 1
                    df = str(
                        pd.read_table(f, nrows=0, skiprows=list(range(index2))).columns[0]
                    ).split(' ')
                    ctr = 0
                    for d in df:
                        if len(d) > 0:
                            data[h3_names[ctr]]['data'].append(d)
                            ctr += 1
                    df = pd.read_csv(
                        f,
                        skiprows=list(range(index2 + 1)),
                        nrows=int(data['n_gates']['data'][-1]) - 1,
                        delim_whitespace=True,
                        names=list(names.keys()),
                    )

                # Add data from the columns to the dictionary
                for n in names:
                    data[n]['data'].append(df[n].to_list())

                # Calculate the range based on number of gates, range to first gate and range between gates
                if len(data['range']['data']) == 0:
                    ranges = float(data['first_range_gate']['data'][-1]) + np.array(
                        range(int(data['n_gates']['data'][-1]) - 1)
                    ) * float(data['range_between_gates']['data'][-1])
                    data['range']['data'] = ranges

                # Calculate a time
                time = dt.datetime(
                    int('20' + data['year']['data'][-1]),
                    1,
                    1,
                    int(data['hour']['data'][-1]),
                    int(data['minute']['data'][-1]),
                    int(data['second']['data'][-1]),
                ) + dt.timedelta(days=int(data['day_of_year']['data'][-1]) - 1)
                data['time']['data'].append(time)

                # Range correct the snr which converts it essentially to an uncalibrated reflectivity
                snr_rc = data['snr']['data'][-1] - 20.0 * np.log10(1.0 / (ranges / 1000.0) ** 2)
                data['reflectivity_uncalibrated']['data'].append(snr_rc)
            except Exception as e:
                # Handle errors, if end of file then continue on, if something else
                # try the next block of data but if it errors another time in this file move on
                if isinstance(e, pd.errors.EmptyDataError) or error_ct > 1:
                    error = True
                else:
                    print(e)
                    pass
                error_ct += 1
            ct += 1

    # Convert dictionary to Dataset
    ds = xr.Dataset().from_dict(data)

    return ds
