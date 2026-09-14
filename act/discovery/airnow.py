import numpy as np
import pandas as pd
import xarray as xr


def get_airnow_forecast(token, date=None, zipcode=None, latlon=None, distance=25):
    """
    This tool will get current AQI forecast values and categories for a
    reporting area by either Zip code or Lat/Lon coordinate.
    https://docs.airnowapi.org/

    Parameters
    ----------
    token : str
        The access token for accessing the AirNowAPI web server
    date : str
        The date of the forecast to be acquired. Format is YYYY-MM-DD.
        Default is None.
    zipcode : str
        The zipcode of the location for the data request.
        If zipcode is not defined then a latlon coordinate must be defined.
    latlon : array
        The latlon coordinate of the location for the data request.
        If latlon is not defined then a zipcode must be defined.
    distance : int
        If no reporting area is associated with the specified zipcode or latlon,
        return a forecast from a nearby reporting area within this distance
        (in miles). Default is 25 miles

    Returns
    -------
    ds : xarray.Dataset
        Returns an Xarray dataset object

    Example
    -------
    act.discovery.get_airnow_forecast(token='XXXXXX', zipcode='60440')

    """

    # The previous forecast/zipCode and forecast/latLong services are retired
    # September 30, 2026. Both are replaced by forecast/current.
    query_url = 'https://www.airnowapi.org/aq/forecast/current/?'

    # Check whether either a zipcode or latlon coordinate is defined.
    if (zipcode is None) and (latlon is None):
        raise NameError("Zipcode or latlon must be defined")

    if zipcode:
        url = query_url + ('format=text/csv' + '&zipCode=' + str(zipcode))
    if latlon:
        url = query_url + (
            'format=text/csv' + '&latitude=' + str(latlon[0]) + '&longitude=' + str(latlon[1])
        )

    # Retain the date argument for backwards compatibility. The new current
    # forecast service can be queried without a date, so only append it when
    # explicitly provided.
    if date is not None:
        url += '&date=' + str(date)

    url += '&distance=' + str(distance) + '&API_KEY=' + str(token)

    df = pd.read_csv(url)

    if 'Aqi' in df.columns:
        df = df.rename(columns={'Aqi': 'AQI'})

    # Convert to xarray dataset object.
    ds = df.to_xarray()

    return ds


def get_airnow_obs(token, date=None, zipcode=None, latlon=None, state=None, distance=25):
    """
    This tool will get current or historical observed AQI values and categories.
    Current observations can be requested by Zip code or Lat/Lon coordinate.
    Historical daily observations are requested by state.
    https://docs.airnowapi.org/

    Parameters
    ----------
    token : str
        The access token for accessing the AirNowAPI web server
    date : str
        The date of the data to be acquired. Format is YYYY-MM-DD.
        Default is None, which will pull the most recent observations.
    zipcode : str
        The zipcode of the location for a current data request.
        If zipcode is not defined then a latlon coordinate must be defined.
    latlon : array
        The latlon coordinate of the location for a current data request.
        If latlon is not defined then a zipcode must be defined.
    state : str
        Two-letter state code used for historical daily observations.
        Required when date is provided.
    distance : int
        For current observations, search for data within this distance
        (in miles). Default is 25 miles.

    Returns
    -------
    ds : xarray.Dataset
        Returns an xarray dataset object

    Example
    -------
    act.discovery.get_airnow_obs(token='XXXXXX', zipcode='60440')
    act.discovery.get_airnow_obs(token='XXXXXX', latlon=[45, -87])
    act.discovery.get_airnow_obs(token='XXXXXX', date='2025-05-01', state='IL')

    """

    query_url = 'https://www.airnowapi.org/aq/observation/'

    if date is None:
        # The previous observation/zipCode/current and
        # observation/latLong/current services are retired September 30, 2026.
        # Both are replaced by observation/current/ziplatlong.
        if (zipcode is None) and (latlon is None):
            raise NameError("Zipcode or latlon must be defined")

        query_url += 'current/ziplatlong/?'

        if zipcode:
            url = query_url + (
                'format=text/csv'
                + '&zipCode='
                + str(zipcode)
                + '&distance='
                + str(distance)
                + '&API_KEY='
                + str(token)
            )

        if latlon:
            url = query_url + (
                'format=text/csv'
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
        # Historical ZIP and Lat/Lon observation services are retired
        # September 30, 2026. The replacement returns historical daily
        # observations for Reporting Areas within a selected state.
        if state is None:
            raise NameError("State must be defined for historical observations")

        url = (
            query_url
            + 'historical/state/?'
            + 'format=text/csv'
            + '&stateCode='
            + str(state)
            + '&startDate='
            + str(date)
            + '&endDate='
            + str(date)
            + '&API_KEY='
            + str(token)
        )
    df = pd.read_csv(url)

    if '' in df.columns:
        df = df.rename(columns={'Aqi': 'AQI'})

    # Convert to xarray.
    ds = df.to_xarray()

    return ds


def get_airnow_bounded_obs(
    token, start_date, end_date, latlon_bnds, parameters='OZONE,PM25', data_type='B', mon_type=0
):
    """
    Get AQI values or data concentrations for a specific date and time range and set of
    parameters within a geographic area of interest.
    https://docs.airnowapi.org/

    Parameters
    ----------
    token : str
        The access token for accessing the AirNowAPI web server
    start_date : str
        The start date and hour (in UTC) of the data request.
        Format is YYYY-MM-DDTHH
    end_date : str
        The end date and hour (in UTC) of the data request.
        Format is YYYY-MM-DDTHH
    latlon_bnds : str
        Lat/Lon bounding box of the area of interest.
        Format is 'minX,minY,maxX,maxY'
    parameters : str
        Parameters to return data for. Options are:
        Ozone, PM25, PM10, CO, NO2, SO2
        Format is 'PM25,PM10'
    mon_type : int
        The type of monitor to be returned. Default is 0
        0-Permanent, 1-Mobile only, 2-Permanent & Mobile
    data_type : char
        The type of data to be returned.
        A-AQI, C-Concentrations, B-AQI & Concentrations

    Returns
    -------
    ds : xarray.Dataset
        Returns an xarray dataset object

    """

    # The hourly aq/data service is not being retired in the 2026 API update.
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

    if 'Aqi' in df.columns:
        df = df.rename(columns={'Aqi': 'AQI'})

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

    # Set up empty data with nans
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
