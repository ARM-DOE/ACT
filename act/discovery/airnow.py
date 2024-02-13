import pandas as pd
import numpy as np
import xarray as xr


def get_airnow_forecast(token, date, zipcode=None, latlon=None, distance=25):
    """
    This tool will get current or historical AQI values and categories for a
    reporting area by either Zip code or Lat/Lon coordinate.
    https://docs.airnowapi.org/

    Parameters
    ----------
    token : str
        The access token for accesing the AirNowAPI web server
    date : str
        The date of the data to be acquired. Format is YYYY-MM-DD
    zipcode : str
        The zipcode of the location for the data request.
        If zipcode is not defined then a latlon coordinate must be defined.
    latlon : array
        The latlon coordinate of the loaction for the data request.
        If latlon is not defined then a zipcode must be defined.
    distance : int
        If no reporting are is associated with the specified zipcode or latlon,
        return a forcast from a nearby reporting area with this distance (in miles).
        Default is 25 miles

    Returns
    -------
    ds : xarray.Dataset
        Returns an Xarray dataset object

    Example
    -------
    act.discovery.get_AirNow_forecast(token='XXXXXX', zipcode='60440', date='2012-05-31')

    """

    # default beginning of the query url
    query_url = 'https://airnowapi.org/aq/forecast/'

    # checking is either a zipcode or latlon coordinate is defined
    # if neither is defined then error is raised
    if (zipcode is None) and (latlon is None):
        raise NameError("Zipcode or latlon must be defined")

    if zipcode:
        url = query_url + (
            'zipcode/?'
            + 'format=text/csv'
            + '&zipCode='
            + str(zipcode)
            + '&date='
            + str(date)
            + '&distance='
            + str(distance)
            + '&API_KEY='
            + str(token)
        )

    if latlon:
        url = query_url + (
            'latLong/?'
            + 'format=text/csv'
            + '&latitude='
            + str(latlon[0])
            + '&longitude='
            + str(latlon[1])
            + '&date='
            + str(date)
            + '&distance='
            + str(distance)
            + '&API_KEY='
            + str(token)
        )

    df = pd.read_csv(url)

    # converting to xarray dataset object
    ds = df.to_xarray()

    return ds


def get_airnow_obs(token, date=None, zipcode=None, latlon=None, distance=25):
    """
    This tool will get current or historical observed AQI values and categories for a
    reporting area by either Zip code or Lat/Lon coordinate.
    https://docs.airnowapi.org/

    Parameters
    ----------
    token : str
        The access token for accesing the AirNowAPI web server
    date : str
        The date of the data to be acquired. Format is YYYY-MM-DD
        Default is None which will pull most recent observations
    zipcode : str
        The zipcode of the location for the data request.
        If zipcode is not defined then a latlon coordinate must be defined.
    latlon : array
        The latlon coordinate of the loaction for the data request.
        If latlon is not defined then a zipcode must be defined.
    distance : int
        If no reporting are is associated with the specified zipcode or latlon,
        return a forcast from a nearby reporting area with this distance (in miles).
        Default is 25 miles

    Returns
    -------
    ds : xarray.Dataset
        Returns an xarray dataset object

    Example
    -------
    act.discovery.get_AirNow_obs(token='XXXXXX', date='2021-12-01', zipcode='60440')
    act.discovery.get_AirNow_obs(token='XXXXXX', latlon=[45,-87])

    """

    # default beginning of the query url
    query_url = 'https://www.airnowapi.org/aq/observation/'

    # checking is either a zipcode or latlon coordinate is defined
    # if neither is defined then error is raised
    if (zipcode is None) and (latlon is None):
        raise NameError("Zipcode or latlon must be defined")

    # setting the observation type to either current or historical based on the date
    if date is None:
        obs_type = 'current'
        if zipcode:
            url = query_url + (
                'zipCode/'
                + str(obs_type)
                + '/?'
                + 'format=text/csv'
                + '&zipCode='
                + str(zipcode)
                + '&distance='
                + str(distance)
                + '&API_KEY='
                + str(token)
            )
        if latlon:
            url = query_url + (
                'latLong/'
                + str(obs_type)
                + '/?'
                + 'format=text/csv'
                + '&latitude='
                + str(latlon[0])
                + '&longitude='
                + str(latlon[1])
                + '&distance='
                + str(distance)
                + '&API_KEY='
                + str(token)
            )
    else:
        obs_type = 'historical'
        if zipcode:
            url = query_url + (
                'zipCode/'
                + str(obs_type)
                + '/?'
                + 'format=text/csv'
                + '&zipCode='
                + str(zipcode)
                + '&date='
                + str(date)
                + 'T00-0000&distance='
                + str(distance)
                + '&API_KEY='
                + str(token)
            )
        if latlon:
            url = query_url + (
                'latLong/'
                + str(obs_type)
                + '/?'
                + 'format=text/csv'
                + '&latitude='
                + str(latlon[0])
                + '&longitude='
                + str(latlon[1])
                + '&date='
                + str(date)
                + 'T00-0000&distance='
                + str(distance)
                + '&API_KEY='
                + str(token)
            )

    df = pd.read_csv(url)

    # converting to xarray
    ds = df.to_xarray()

    return ds


def get_airnow_bounded_obs(
    token, start_date, end_date, latlon_bnds, parameters='OZONE,PM25', data_type='B', mon_type=0
):
    """
    Get AQI values or data concentrations for a specific date and time range and set of
    parameters within a geographic area of intrest
    https://docs.airnowapi.org/

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
        Ozone, PM25, PM10, CO, NO2, SO2
        Format is 'PM25,PM10'
    mon_type : int
        The type of monitor to be returned. Default is 0
        0-Permanent, 1-Mobile onlt, 2-Permanent & Mobile
    data_type : char
        The type of data to be returned.
        A-AQI, C-Concentrations, B-AQI & Concentrations

    Returns
    -------
    ds : xarray.Dataset
        Returns an xarray dataset object

    """

    verbose = 1
    inc_raw_con = 1

    url = (
        'https://www.airnowapi.org/aq/data/?startDate='
        + str(start_date)
        + '&endDate='
        + str(end_date)
        + '&parameters='
        + str(parameters)
        + '&BBOX='
        + str(latlon_bnds)
        + '&dataType='
        + str(data_type)
        + '&format=text/csv'
        + '&verbose='
        + str(verbose)
        + '&monitorType='
        + str(mon_type)
        + '&includerawconcentrations='
        + str(inc_raw_con)
        + '&API_KEY='
        + str(token)
    )

    # Set Column names
    names = [
        'latitude',
        'longitude',
        'time',
        'parameter',
        'concentration',
        'unit',
        'raw_concentration',
        'AQI',
        'category',
        'site_name',
        'site_agency',
        'aqs_id',
        'full_aqs_id',
    ]

    # Read data into CSV
    df = pd.read_csv(url, names=names)

    # Each line is a different time or site or variable so need to parse out
    sites = df['site_name'].unique()
    times = df['time'].unique()
    variables = list(df['parameter'].unique()) + ['AQI', 'category', 'raw_concentration']
    latitude = [list(df['latitude'].loc[df['site_name'] == s])[0] for s in sites]
    longitude = [list(df['longitude'].loc[df['site_name'] == s])[0] for s in sites]
    aqs_id = [list(df['aqs_id'].loc[df['site_name'] == s])[0] for s in sites]

    # Set up the dataset ahead of time
    ds = xr.Dataset(
        data_vars={
            'latitude': (['sites'], latitude),
            'longitude': (['sites'], longitude),
            'aqs_id': (['sites'], aqs_id),
        },
        coords={'time': (['time'], times), 'sites': (['sites'], sites)},
    )

    # Set up emtpy data with nans
    data = np.empty((len(variables), len(times), len(sites)))
    data[:] = np.nan

    # For each variable, pull out the data from specific sites and times
    for v in range(len(variables)):
        for t in range(len(times)):
            for s in range(len(sites)):
                if variables[v] in ['AQI', 'category', 'raw_concentration']:
                    result = df.loc[(df['time'] == times[t]) & (df['site_name'] == sites[s])]
                    if len(result[variables[v]]) > 0:
                        data[v, t, s] = list(result[variables[v]])[0]
                        atts = {'units': ''}
                else:
                    result = df.loc[
                        (df['time'] == times[t])
                        & (df['site_name'] == sites[s])
                        & (df['parameter'] == variables[v])
                    ]
                    if len(result['concentration']) > 0:
                        data[v, t, s] = list(result['concentration'])[0]
                        atts = {'units': list(result['unit'])[0]}

        # Add variables to the dataset
        ds[variables[v]] = xr.DataArray(data=data[v, :, :], dims=['time', 'sites'], attrs=atts)

    times = pd.to_datetime(times)
    ds = ds.assign_coords({'time': times})
    return ds
