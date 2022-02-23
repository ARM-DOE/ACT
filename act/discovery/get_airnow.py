import pandas as pd
import os
from datetime import datetime
import urllib


def get_AirNow_forecast(token, date, format='text/csv',
                        zipcode=None, latlon=None, distance=25):
    """
    This tool will get current or historical AQI values and categories for a
    reporting area by either Zip code or Lat/Lon coordinate.
    Parameters
    ----------
    token : str
        The access token for accesing the AirNowAPI web server
    date : str
        The date of the data to be acquired. Format is YYYY-MM-DD
    format : str
        The format of the data returned. Default is text/csv.
        Other options include application/json or application/xml
    zipcode : str
        The zipcode of the location for the data request.
        If zipcode is not defined then a latlon coordinate must be defined.
    latlon : array
        The latlon coordinate of the loaction for the data request.
        If latlon is not defined then a zipcode must be defined.
    distance : int
        If no reporting are is associated with the specified zipcode or latlon,
        return a forcast from a nearby reporting area with this distance
        (in miles). Default is 25 miles
    Returns
    -------
    df : xarray dataset
        Returns an xarray data object
    Example
    -------
    act.discovery.get_AirNow_forecast(token='XXXXXX', zipcode='60440',
                                      date='2012-05-31', format='text/csv')
    """

    # Default beginning of the query url
    query_url = ('https://airnowapi.org/aq/forecast/')

    # Checking is either a zipcode or latlon coordinate is defined
    # If neither is defined then error is raised
    if (zipcode is None) and (latlon is None):
        raise NameError("Zipcode or latlon must be defined")

    if zipcode:
        url = (query_url + ('zipcode/?' + 'format=' + str(format) +
                            '&zipCode=' + str(zipcode) +
                            '&date=' + str(date) +
                            '&distance=' + str(distance) +
                            '&API_KEY=' + str(token)))

    if latlon:
        url = (query_url + ('latLong/?' + 'format=' + str(format) +
                            '&latitude=' + str(latlon[0]) + '&longitude=' +
                            str(latlon[1]) + '&date=' + str(date) +
                            '&distance=' + str(distance) +
                            '&API_KEY=' + str(token)))

    if format == 'text/csv':
        df = pd.read_csv(url)

    if format == 'application/json':
        df = pd.read_json(url)

    if format == 'application/xml':
        df = pd.read_xml(url)

    # Converting to xarray object
    df = df.to_xarray()

    return df


def get_AirNow_obs(token, format='text/csv', date=None,
                   zipcode=None, latlon=None, distance=25):
    """
    This tool will get current or historical observed AQI values
    and categories for a reporting area by either Zip code or
    Lat/Lon coordinate.
    Parameters
    ----------
    token : str
        The access token for accesing the AirNowAPI web server
    date : str
        The date of the data to be acquired. Format is YYYY-MM-DD
        Default is None which will pull most recent observations
    format : str
        The format of the data returned. Default is text/csv.
        Other options include application/json or application/xml
    zipcode : str
        The zipcode of the location for the data request.
        If zipcode is not defined then a latlon coordinate must be defined.
    latlon : array
        The latlon coordinate of the loaction for the data request.
        If latlon is not defined then a zipcode must be defined.
    distance : int
        If no reporting are is associated with the specified zipcode or latlon,
        return a forcast from a nearby reporting area with this
        distance (in miles). Default is 25 miles
    Returns
    -------
    df : xarray dataset
        Returns an xarray data object
    Example
    -------
    act.discovery.get_AirNow_obs(token='XXXXXX',
                                 date='2021-12-01',
                                 zipcode='60440')

    act.discovery.get_AirNow_obs(token='XXXXXX',
                                 latlon=[45,-87])
    """

    # Default beginning of the query url
    query_url = ('https://www.airnowapi.org/aq/observation/')

    # Checking is either a zipcode or latlon coordinate is defined
    # If neither is defined then error is raised
    if (zipcode is None) and (latlon is None):
        raise NameError("Zipcode or latlon must be defined")

    # Setting the observation type to either current or
    # historical based on the date
    if date is None:
        obs_type = 'current'
        if zipcode:
            url = (query_url + ('zipCode/' + str(obs_type) + '/?' + 'format=' +
                                str(format) + '&zipCode=' + str(zipcode) +
                                '&distance=' + str(distance) + '&API_KEY=' +
                                str(token)))
        if latlon:
            url = (query_url + ('latLong/' + str(obs_type) + '/?' + 'format=' +
                                str(format) + '&latitude=' + str(latlon[0]) +
                                '&longitude=' + str(latlon[1]) + '&distance=' +
                                str(distance) + '&API_KEY=' + str(token)))
    else:
        obs_type = 'historical'
        if zipcode:
            url = (query_url + ('zipCode/' + str(obs_type) + '/?' + 'format=' +
                                str(format) + '&zipCode=' + str(zipcode) +
                                '&date=' + str(date) + 'T00-0000&distance=' +
                                str(distance) + '&API_KEY=' + str(token)))
        if latlon:
            url = (query_url + ('latLong/' + str(obs_type) + '/?' + 'format=' +
                                str(format) + '&latitude=' + str(latlon[0]) +
                                '&longitude=' + str(latlon[1]) + '&date=' +
                                str(date) + 'T00-0000&distance=' +
                                str(distance) + '&API_KEY=' + str(token)))

    if format == 'text/csv':
        df = pd.read_csv(url)

    if format == 'application/json':
        df = pd.read_json(url)

    if format == 'application/xml':
        df = pd.read_xml(url)

    # Converting to xarray
    df = df.to_xarray()

    return df


def get_AirNow(token, start_date, end_date, latlon_bnds, parameters, data_type,
               format='text/csv', ext='csv', inc_raw_con=False, mon_type=0,
               output=None, verbose=True):
    """
    Get AQI values or data concentrations for a specific date and time range
    and set of parameters within a geographic area of intrest
    Parameters
    ----------
    token : str
        The access token for accesing the AirNowAPI web server
    start_date : str
        The start date and hour (in UTC) of the data request.
        Format is YYYY-MM-DDTHH
    end_date : str
        The end date and hour (in UTC) of the data request.
        Format is YYYY-MM-DDTHH
    latlon_bnds : str
        Lat/Lon bounding box of the area of intrest.
        Format is 'minX,minY,maxX,maxY'
    parameters : str
        Parameters to return data for. Options are:
        Ozone (O3), PM2.5 (PM25), PM10 (PM10), CO (co),
        NO2 (no2), and SO2 (so2)
        Format is 'PM25,PM10'
    mon_type : int
        The type of monitor to be returned. Default is 0
        0-Permanent, 1-Mobile onlt, 2-Permanent & Mobile
    data_type : char
        The type of data to be returned.
        A-AQI, C-Concentrations, B-AQI & Concentrations
    format : str
        Format of the data type to be returned. Default is text/csv
        Other options include: applications/json, application/kml,
                               application/xml
    ext : str
        Extention type for when saving data. Default is csv
        other options are json, kml, and xml
    verbose : bool
        provides additional site information like Site Name, Agency Name,
        AQS ID, and Full AQS ID
    inc_raw_con : bool
        Adds additional field that contains the raw concentration.
        For CO, NO2, and SO2 these are the same as the concentration.
        For O3, PM2.5, and PM10 these are the raw hourly measured
        concentrations by the instrument. Units are the same as those
        specified in the units field
    output : str
        The output directory for the data to be saved. If no output path set
        then data will be saved in current working directory
    Returns
    -------
    file : str
        Returns path of the query url and directory path of the saved file
    """

    if verbose:
        verbose = 1
    else:
        verbose = 0

    if inc_raw_con:
        inc_raw_con = 1
    else:
        inc_raw_con = 0

    query_url = ('https://www.airnowapi.org/aq/data/?startDate=' +
                 str(start_date) + '&endDate=' + str(end_date) +
                 '&parameters=' + str(parameters) +
                 '&BBOX=' + str(latlon_bnds) + '&dataType=' + str(data_type) +
                 '&format=' + str(format) + '&verbose=' + str(verbose) +
                 '&monitorType=' + str(mon_type) +
                 '&includerawconcentrations=' +
                 str(inc_raw_con) + '&API_KEY=' + str(token))

    start_date_time = datetime.strptime(
        start_date, '%Y-%m-%dT%H').strftime('%Y%m%dT%H')
    end_date_time = datetime.strptime(
        end_date, '%Y-%m-%dT%H').strftime('%Y%m%dT%H')

    try:
        # Requesting AirNowAPI data
        download_file_name = ('AirNowAPI' + start_date_time + '_' +
                              end_date_time + '.' + ext)

        # Get current working dir if no output path is set
        if output:
            output_dir = os.path.join(output)
        else:
            output_dir = os.getcwd()

        download_file = os.path.join(output_dir, download_file_name)

        # Perform the airnow API data request
        urllib.request.urlretrieve(query_url, download_file)

        # Download complete
        print(f'Download URL: {query_url}')
        print(f'Download File: {download_file}')
    except Exception as e:
        print(f'Unable to perform AirNowAPI request. {e}')
