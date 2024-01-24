"""
Module containing utilities for geographic calculations,
including solar calculations

"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytz
from skyfield import almanac
from skyfield.api import load, load_file, wgs84

from act.utils.data_utils import convert_units
from act.utils.datetime_utils import datetime64_to_datetime

skyfield_bsp_file = str(Path(Path(__file__).parent, 'conf', 'de421.bsp'))


def destination_azimuth_distance(lat, lon, az, dist, dist_units='m'):
    """
    This procedure will calculate a destination lat/lon from
    an initial lat/lon and azimuth and distance.

    Parameters
    ----------
    lat : float
        Initial latitude.
    lon : float
        Initial longitude.
    az : float
        Azimuth in degrees.
    dist : float
        Distance
    dist_units : str
        Units for dist

    Returns
    -------
    lat2 : float
        Latitude of new point in degrees
    lon2 : float
        Longitude of new point in degrees

    """
    # Volumetric Mean Radius of Earth in km
    R = 6378.0

    # Convert az to radian
    brng = np.radians(az)

    # Convert distance to km
    d = convert_units(dist, dist_units, 'km')

    # Convert lat/lon to radians
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Using great circle equations
    lat2 = np.arcsin(np.sin(lat) * np.cos(d / R) + np.cos(lat) * np.sin(d / R) * np.cos(brng))
    lon2 = lon + np.arctan2(
        np.sin(brng) * np.sin(d / R) * np.cos(lat),
        np.cos(d / R) - np.sin(lat) * np.sin(lat2),
    )

    return np.degrees(lat2), np.degrees(lon2)


def add_solar_variable(ds, latitude=None, longitude=None, solar_angle=0.0, dawn_dusk=False):
    """
    Add variable to the dataset to denote night (0) or sun (1). If dawk_dusk is True
    will also return dawn (2) and dusk (3). If at a high latitude and there's sun, will
    label twilight as dawn; if dark{2}, will label twilight as dusk(3).

    Parameters
    ----------
    ds : xarray.Dataset
        ACT Xarray dataset
    latitude : str
        Latitude variable name, default will look for matching variables in
        the dataset.
    longitude : str
        Longitude variable name, default will look for matching variables in
        the dataset.
    solar_angle : float
        Number of degress to use for dawn/dusk calculations
    dawn_dusk : boolean
         If set to True, will add values 2 (dawn) and 3 (dusk) to the solar variable

    Returns
    -------
    ds : xarray.Dataset
        Xarray dataset containing sun and night flag.
    """
    variables = list(ds.keys())

    # Get coordinate variables
    if latitude is None:
        latitude = [s for s in variables if 'latitude' in s]
        if len(latitude) == 0:
            latitude = [s for s in variables if 'lat' in s]
        if len(latitude) == 0:
            raise ValueError('Latitude variable not set and could not be discerned from the data')

    if longitude is None:
        longitude = [s for s in variables if 'longitude' in s]
        if len(longitude) == 0:
            longitude = [s for s in variables if 'lon' in s]
        if len(longitude) == 0:
            raise ValueError('Longitude variable not set and could not be discerned from the data')

    # Get lat/lon variables
    lat = ds[latitude[0]].values
    lon = ds[longitude[0]].values

    # Loop through each time to ensure that the sunrise/set calcuations
    # are correct for each time and lat/lon if multiple
    results = is_sun_visible(
        latitude=lat, longitude=lon, date_time=ds['time'].values, dawn_dusk=dawn_dusk
    )

    # Set longname
    longname = 'Daylight indicator; 0-Night; 1-Sun'

    if dawn_dusk is False:
        results = results * 1
    else:
        # If dawn_dusk is True, add 2 more indicators
        longname += '; 2-Dawn; 3-Dusk; 4-Twilight'
        dark_ind = np.where(results == 0)[0]
        twil_ind = np.where((results > 0) & (results < 4))[0]
        sun_ind = np.where(results == 4)[0]

        if len(sun_ind) == 0:
            results[twil_ind] = 3
        elif len(dark_ind) == 0:
            results[twil_ind] = 2
            results[sun_ind] = 1
        else:
            # Set Dawn between dark and sun
            if dark_ind[-1] < sun_ind[0]:
                dawn_ind = list(range(dark_ind[-1], sun_ind[0]))
            else:
                dawn_ind = list(range(dark_ind[-1], len(results))) + list(range(0, sun_ind[0]))
            results[dawn_ind] = 2

            # Set Dusk between sun and dark
            if sun_ind[-1] < dark_ind[0]:
                dusk_ind = list(range(sun_ind[-1], dark_ind[0]))
            else:
                dusk_ind = list(range(sun_ind[-1], len(results))) + list(range(0, dark_ind[0]))
            results[dusk_ind] = 3
            results[sun_ind] = 1

    # Add results to the dataset and return
    ds['sun_variable'] = (
        'time',
        np.array(results),
        {'long_name': longname, 'units': ' '},
    )

    return ds


def get_solar_azimuth_elevation(
    latitude=None,
    longitude=None,
    time=None,
    library='skyfield',
    temperature_C='standard',
    pressure_mbar='standard',
):
    """
    Calculate solar azimuth, elevation and solar distance.


    Parameters
    ----------
    latitude : int, float
        Latitude in degrees north positive. Must be a scalar.
    longitude : int, float
        Longitude in degrees east positive. Must be a scalar.
    time : datetime.datetime, numpy.datetime64, list, numpy.array
        Time in UTC. May be a scalar or vector. datetime must be timezone aware.
    library : str
        Library to use for making calculations. Options include ['skyfield']
    temperature_C : string or list of float
        If library is 'skyfield' the temperature in degrees C at the surface for
        atmospheric compensation of the positon of the sun. Set to None for no
        compensation or 'standard' for standard model with a standard temperature.
    pressure_mbar : string or list of float
        If library is 'skyfield' the pressure in milibars at the surface for
        atmospheric compensation of the positon of the sun. Set to None for no
        compensation or 'standard' for standard model with a standard pressure.

    Returns
    -------
    result : tuple of float
        Values returned are a tuple of elevation, azimuth and distance. Elevation and
        azimuth are in degrees, with distance in Astronomical Units.

    """

    # result = {'elevation': None, 'azimuth': None, 'distance': None}
    result = (None, None, None)

    if library == 'skyfield':
        planets = load_file(skyfield_bsp_file)
        earth, sun = planets['earth'], planets['sun']

        if isinstance(time, datetime) and time.tzinfo is None:
            time = time.replace(tzinfo=pytz.UTC)

        if isinstance(time, (list, tuple)) and time[0].tzinfo is None:
            time = [ii.replace(tzinfo=pytz.UTC) for ii in time]

        if type(time).__module__ == np.__name__ and np.issubdtype(time.dtype, np.datetime64):
            time = time.astype('datetime64[s]').astype(int)
            if time.size > 1:
                time = [datetime.fromtimestamp(tm, timezone.utc) for tm in time]
            else:
                time = [datetime.fromtimestamp(time, timezone.utc)]

        if not isinstance(time, (list, tuple, np.ndarray)):
            time = [time]

        ts = load.timescale()
        t = ts.from_datetimes(time)
        location = earth + wgs84.latlon(latitude, longitude)
        astrometric = location.at(t).observe(sun)
        alt, az, distance = astrometric.apparent().altaz(
            temperature_C=temperature_C, pressure_mbar=pressure_mbar
        )
        result = (alt.degrees, az.degrees, distance.au)
        planets.close()

    return result


def get_sunrise_sunset_noon(
    latitude=None, longitude=None, date=None, library='skyfield', timezone=False
):
    """
    Calculate sunrise, sunset and local solar noon times.

    Parameters
    ----------
    latitude : int, float
        Latitude in degrees north positive. Must be a scalar.
    longitude : int, float
        Longitude in degrees east positive. Must be a scalar.
    date : (datetime.datetime, numpy.datetime64, list of datetime.datetime,
            numpy.array of numpy.datetime64, string, list of string)
        Date(s) to return sunrise, sunset and noon times spaning the first date to last
        date if more than one provided. May be a scalar or vector. If entered as a string must follow
        YYYYMMDD format.
    library : str
        Library to use for making calculations. Options include ['skyfield']
    timezone : boolean
        Have timezone with datetime.

    Returns
    -------
    result : tuple of three numpy.array
        Tuple of three values sunrise, sunset, noon. Values will be a list.
        If no values can be calculated will return empty list. If the date is within
        polar night will return empty lists. If spans the transition to polar day
        will return previous sunrise or next sunset outside of date range provided.
    """
    sunrise, sunset, noon = np.array([]), np.array([]), np.array([])

    if library == 'skyfield':
        ts = load.timescale()
        eph = load_file(skyfield_bsp_file)
        sf_dates = []

        # Parse datetime object
        if isinstance(date, datetime):
            if date.tzinfo is None:
                sf_dates = [date.replace(tzinfo=pytz.UTC)]
            else:
                sf_dates = [date]

        if isinstance(date, (list, tuple)) and isinstance(date[0], datetime):
            if date[0].tzinfo is not None:
                sf_dates = date
            else:
                sf_dates = [ii.replace(tzinfo=pytz.UTC) for ii in date]

        # Parse string date
        if isinstance(date, str):
            sf_dates = [datetime.strptime(date, '%Y%m%d').replace(tzinfo=pytz.UTC)]

        # Parse list of string dates
        if isinstance(date, (list, tuple)) and isinstance(date[0], str):
            sf_dates = [datetime.strptime(dt, '%Y%m%d').replace(tzinfo=pytz.UTC) for dt in date]

        # Convert datetime64 to datetime
        if type(date).__module__ == np.__name__ and np.issubdtype(date.dtype, np.datetime64):
            sf_dates = datetime64_to_datetime(date)
            sf_dates = [ii.replace(tzinfo=pytz.UTC) for ii in sf_dates]

        # Function for calculating solar noon
        # Convert location into skyfield location object
        location = wgs84.latlon(latitude, longitude)
        # Set up function to indicate calculating locatin of Sun from Earth
        f = almanac.meridian_transits(eph, eph['Sun'], location)
        # Set up dates to be start of day and end of day so have a range
        t0 = sf_dates[0]
        t0 = t0.replace(hour=0, minute=0, second=0)
        t1 = sf_dates[-1]
        t1 = t1.replace(hour=23, minute=59, second=59)
        # Convert times from datetime to skyfild times
        t0 = ts.from_datetime(t0)
        t1 = ts.from_datetime(t1)
        # Calculate Meridian Transits. n contains times and x contains 1 and 0's
        # indicating when transit time is above or below location.
        n, x = almanac.find_discrete(t0, t1, f)

        # Determine if time is during daylight
        f = almanac.sunrise_sunset(eph, location)
        sun_up = f(n)

        # Filter out times when sun is below location or in polar night
        n = n[(x == 1) & sun_up]
        noon = n.utc_datetime()
        if noon.size == 0:
            return sunrise, sunset, noon

        # Calcuate sunrise and sunset times. Calcuate over range 12 less than minimum
        # noon time and 12 hours greater than maximum noon time.
        t0 = min(noon) - timedelta(hours=12)
        t1 = max(noon) + timedelta(hours=12)
        t0 = ts.from_datetime(t0)
        t1 = ts.from_datetime(t1)
        f = almanac.sunrise_sunset(eph, location)
        t, y = almanac.find_discrete(t0, t1, f)
        times = t.utc_datetime()
        sunrise = times[y == 1]
        sunset = times[y == 0]

        # Fill in sunrise and sunset if asked to during polar day
        if len(noon) > 0 and (y.size == 0 or len(sunrise) != len(sunset)):
            days = 200
            t0 = min(noon) - timedelta(days=days)
            t1 = max(noon) + timedelta(days=days)
            t0 = ts.from_datetime(t0)
            t1 = ts.from_datetime(t1)
            t, yy = almanac.find_discrete(t0, t1, f)
            times = t.utc_datetime()

            # If first time is sunset and/or last time is sunrise filter
            # from times
            if yy[0] == 0:
                yy = yy[1:]
                times = times[1:]
            if yy[-1] == 1:
                yy = yy[:-1]
                times = times[:-1]

            # Extract sunrise times
            temp_sunrise = times[yy == 1]
            # Extract sunset times
            temp_sunset = times[yy == 0]
            # Look for index closest to first noon time to get the time of last sunrise
            # since we are in polar day.
            diff = temp_sunrise - min(noon)
            sunrise_index = np.max(np.where(diff < timedelta(seconds=1)))
            # Look for index closest to last noon time to get the time of first sunset
            # since we are in polar day.
            diff = max(noon) - temp_sunset
            sunset_index = np.min(np.where(diff < timedelta(seconds=1))) + 1
            sunrise = temp_sunrise[sunrise_index:sunset_index]
            sunset = temp_sunset[sunrise_index:sunset_index]

        eph.close()

    if timezone is False:
        for ii in range(0, sunset.size):
            sunrise[ii] = sunrise[ii].replace(tzinfo=None)
            sunset[ii] = sunset[ii].replace(tzinfo=None)
        for ii in range(0, noon.size):
            noon[ii] = noon[ii].replace(tzinfo=None)

    return sunrise, sunset, noon


def is_sun_visible(latitude=None, longitude=None, date_time=None, dawn_dusk=False):
    """
    Determine if sun is above horizon at for a list of times.

    Parameters
    ----------
    latitude : int, float
        Latitude in degrees north positive. Must be a scalar.
    longitude : int, float
        Longitude in degrees east positive. Must be a scalar.
    date_time : datetime.datetime, numpy.array.datetime64, list of datetime.datetime
        Datetime with timezone, datetime with no timezone in UTC, or numpy.datetime64
        format in UTC. Can be a single datetime object or list of datetime objects.
    dawn_dusk : boolean
        If set to True, will use skyfields dark_twilight_day function to calculate sun up
        Returns a list of int's instead of boolean.
        0 - Dark of Night
        1 - Astronomical Twilight
        2 - Nautical Twilight
        3 - Civil Twilight
        4 - Sun Is Up

    Returns
    -------
    result : list
        List matching size of date_time containing True/False if sun is above horizon.
    """
    sf_dates = None

    # Check if datetime object is scalar and if has no timezone.
    if isinstance(date_time, datetime):
        if date_time.tzinfo is None:
            sf_dates = [date_time.replace(tzinfo=pytz.UTC)]
        else:
            sf_dates = [date_time]

    # Check if datetime objects in list have timezone. If not add.
    if isinstance(date_time, (list, tuple)) and isinstance(date_time[0], datetime):
        if isinstance(date_time[0], datetime) and date_time[0].tzinfo is not None:
            sf_dates = date_time
        else:
            sf_dates = [ii.replace(tzinfo=pytz.UTC) for ii in date_time]

    # Convert datetime64 to datetime with timezone.
    if type(date_time).__module__ == np.__name__ and np.issubdtype(date_time.dtype, np.datetime64):
        sf_dates = datetime64_to_datetime(date_time)
        sf_dates = [ii.replace(tzinfo=pytz.UTC) for ii in sf_dates]

    if sf_dates is None:
        raise ValueError(
            'The date_time values entered into is_sun_visible() ' 'do not match input types.'
        )

    ts = load.timescale()
    eph = load_file(skyfield_bsp_file)

    t0 = ts.from_datetimes(sf_dates)
    location = wgs84.latlon(latitude, longitude)
    if dawn_dusk:
        f = almanac.dark_twilight_day(eph, location)
    else:
        f = almanac.sunrise_sunset(eph, location)

    sun_up = f(t0)

    eph.close()

    return sun_up
