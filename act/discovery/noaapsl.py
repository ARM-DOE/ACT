"""
Function for downloading data from NOAA PSL Profiler Network

"""
from datetime import datetime
import pandas as pd
import os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


def download_noaa_psl_data(
    site=None, instrument=None, startdate=None, enddate=None, hour=None, output=None
):
    """
    Function to download data from the NOAA PSL Profiler Network Data Library
    https://psl.noaa.gov/data/obs/datadisplay/

    Parameters
    ----------
    site : str
        3 letter NOAA site identifier. Required variable
    instrument : str
        Name of the dataset to download.  Options currently include (name prior to -). Required variable
            'Parsivel' - Parsivel disdrometer data
            'Pressure', 'Datalogger', 'Net Radiation', 'Temp/RH', 'Solar Radiation' - Surface meteorology/radiation data
            'Tipping Bucket', 'TBRG', 'Wind Speed', 'Wind Direction' - Surface meteorology/radiation data
            'Wind Speed and Direction' - Surface meteorology/radiation data
            'GpsTrimble' - GPS Trimble water vapor data
            'Radar S-band Moment' - 3 GHz Precipitation Profiler moment data
            'Radar S-band Bright Band' - 3 GHz Precipitation Profiler bright band data
            '449RWP Bright Band' - 449 MHz Wind Profiler bright band data
            '449RWP Wind' - 449 MHz Wind Profiler wind data
            '449RWP Sub-Hour Wind' - 449 MHz Wind Profiler sub-hourly wind data
            '449RWP Sub-Hour Temp' - 449 MHz Wind Profiler sub-hourly temperature data
            '915RWP Wind' - 915 MHz Wind Profiler wind data
            '915RWP Temp' - 915 MHz Wind Profiler temperature data
            '915RWP Sub-Hour Wind' - 915 MHz Wind Profiler sub-hourly wind data
            '915WP Sub-Hour Temp' - 915 MHz Wind Profiler sub-hourly temperature data
            'Radar FMCW Moment' - FMCW Radar moments data
            'Radar FMCW Bright Band' - FMCW Radar bright band data
    startdate : str
        The start date of the data to acquire. Format is YYYYMMDD. Required variable
    enddate : str
        The end date of the data to acquire. Format is YYYYMMDD
    hour : str
        Two digit hour of file to dowload if wanting a specific time
    output : str
        The output directory for the data. Set to None to make a folder in the
        current working directory with the same name as *datastream* to place
        the files in.

    Returns
    -------
    files : list
        Returns list of files retrieved

    """

    if (site is None) or (instrument is None) or (startdate is None):
        raise ValueError('site, instrument, and startdate need to be set')

    datastream = site + '_' + instrument.replace(' ', '_')
    # Convert dates to day of year (doy) for NOAA folder structure
    s_doy = datetime.strptime(startdate, '%Y%m%d').timetuple().tm_yday
    year = datetime.strptime(startdate, '%Y%m%d').year
    if enddate is None:
        enddate = startdate
    e_doy = datetime.strptime(enddate, '%Y%m%d').timetuple().tm_yday

    # Set base URL
    url = 'https://downloads.psl.noaa.gov/psd2/data/realtime/'

    # Set list of strings that all point to the surface meteorology dataset
    met_ds = [
        'Pressure',
        'Datalogger',
        'Net Radiation',
        'Temp/RH',
        'Solar Radiation',
        'Tipping Bucket',
        'TBRG',
        'Wind Speed',
        'Wind Direction',
        'Wind Speed and Direction',
    ]

    # Add to the url depending on which instrument is requested
    if 'Parsivel' in instrument:
        url += 'DisdrometerParsivel/Stats/'
    elif any([d in instrument for d in met_ds]):
        url += 'CsiDatalogger/SurfaceMet/'
    elif 'GpsTrimble' in instrument:
        url += 'GpsTrimble/WaterVapor/'
    elif 'Radar S-band Moment' in instrument:
        url += 'Radar3000/PopMoments/'
    elif 'Radar S-band Bright Band' in instrument:
        url += 'Radar3000/BrightBand/'
    elif '449RWP Bright Band' in instrument:
        url += 'Radar449/BrightBand/'
    elif '449RWP Wind' in instrument:
        url += 'Radar449/WwWind/'
    elif '449RWP Sub-Hour Wind' in instrument:
        url += 'Radar449/WwWindSubHourly/'
    elif '449RWP Sub-Hour Temp' in instrument:
        url += 'Radar449/WwTempSubHourly/'
    elif '915RWP Wind' in instrument:
        url += 'Radar915/WwWind/'
    elif '915RWP Temp' in instrument:
        url += 'Radar915/WwTemp/'
    elif '915RWP Sub-Hour Wind' in instrument:
        url += 'Radar915/WwWindSubHourly/'
    elif '915RWP Sub-Hour Temp' in instrument:
        url += 'Radar915/WwTempSubHourly/'
    elif 'Radar FMCW Moment' in instrument:
        url += 'RadarFMCW/PopMoments/'
    elif 'Radar FMCW Bright Band' in instrument:
        url += 'RadarFMCW/BrightBand/'
    else:
        raise ValueError('Instrument not supported')

    # Construct output directory
    if output:
        # Output files to directory specified
        output_dir = os.path.join(output)
    else:
        # If no folder given, add datastream folder
        # to current working dir to prevent file mix-up
        output_dir = os.path.join(os.getcwd(), datastream)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set up doy ranges, taking into account changes for a new year
    prev_doy = 0
    if e_doy < s_doy:
        r = list(range(s_doy, 366)) + list(range(1, e_doy + 1))
    else:
        r = list(range(s_doy, e_doy + 1))

    # Set filename variable to return
    filenames = []

    # Loop through each doy in range
    for doy in r:
        # if the previous day is greater than current, assume a new year
        # i.e. 365 -> 001
        if prev_doy > doy:
            year += 1
        # Add site, year, and 3-digit day to url
        new_url = url + site + '/' + str(year) + '/' + str(doy).zfill(3) + '/'

        # User pandas to get a list of filenames to download
        # Exclude the first and last records which are "parent directory" and "nan"
        files = pd.read_html(new_url, skiprows=[1])[0]['Name']
        files = list(files[1:-1])

        # Write each file out to a file with same name as online
        for f in files:
            if hour is not None:
                if (str(doy).zfill(3) + str(hour)) not in f and (
                    str(doy).zfill(3) + '.' + str(hour)
                ) not in f:
                    continue
            output_file = os.path.join(output_dir, f)
            try:
                print('Downloading ' + f)
                with open(output_file, 'wb') as open_bytes_file:
                    open_bytes_file.write(urlopen(new_url + f).read())
                filenames.append(output_file)
            except Exception:
                pass
        prev_doy = doy

    return filenames
