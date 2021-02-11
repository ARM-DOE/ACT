"""
act.utils.geo_utils
--------------------

Module containing utilities for geographic calculations,
including solar calculations

"""

import numpy as np
import pandas as pd
import astral
from datetime import datetime, timezone, timedelta
from skyfield.api import wgs84, load, N, W
from skyfield import almanac
import re
import dateutil.parser
import pytz
from pathlib import Path
from act.utils.datetime_utils import datetime64_to_datetime
from act.utils.data_utils import convert_units
try:
    from astral import Observer
    from astral import sun
    ASTRAL = True
except ImportError:
    ASTRAL = False

skyfield_bsp_file = str(Path(Path(__file__).parent, "conf", "de421.bsp"))

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
    R = 6378.

    # Convert az to radian
    brng = np.radians(az)

    # Convert distance to km
    d = convert_units(dist, dist_units, 'km')

    # Convert lat/lon to radians
    lat = np.radians(lat)
    lon = np.radians(lon)

    # Using great circle equations
    lat2 = np.arcsin(np.sin(lat) * np.cos(d / R) +
                     np.cos(lat) * np.sin(d / R) * np.cos(brng))
    lon2 = lon + np.arctan2(np.sin(brng) * np.sin(d / R) * np.cos(lat),
                            np.cos(d / R) - np.sin(lat) * np.sin(lat2))

    return np.degrees(lat2), np.degrees(lon2)


def add_solar_variable(obj, latitude=None, longitude=None, solar_angle=0., dawn_dusk=False):
    """
    Calculate solar times depending on location on earth.

    Astral 2.2 is recommended for best performance and for the dawn/dusk feature as it
    seems like the dawn calculations are wrong with earlier versions.

    Parameters
    ----------
    obj : xarray dataset
        ACT object
    latitude : str
        Latitude variable name, default will look for matching variables in object
    longitude : str
        Longitude variable name, default will look for matching variables in object
    solar_angle : float
        Number of degress to use for dawn/dusk calculations
    dawn_dusk : boolean
         If set to True, will add values 2 (dawn) and 3 (dusk) to the solar variable

    Returns
    -------
    obj : xarray dataset
        Xarray object
    """

    variables = list(obj.keys())

    # Get coordinate variables
    if latitude is None:
        latitude = [s for s in variables if "latitude" in s]
        if len(latitude) == 0:
            latitude = [s for s in variables if "lat" in s]
        if len(latitude) == 0:
            raise ValueError("Latitude variable not set and could not be discerned from the data")

    if longitude is None:
        longitude = [s for s in variables if "longitude" in s]
        if len(longitude) == 0:
            longitude = [s for s in variables if "lon" in s]
        if len(longitude) == 0:
            raise ValueError("Longitude variable not set and could not be discerned from the data")

    # Get lat/lon variables
    lat = obj[latitude[0]].values
    lon = obj[longitude[0]].values

    # Set the the number of degrees the sun must be below the horizon
    # for the dawn/dusk calculation. Need to do this so when the calculation
    # sends an error it is not going to be an inacurate switch to setting
    # the full day.
    if ASTRAL:
        astral.solar_depression = solar_angle
    else:
        a = astral.Astral()
        a.solar_depression = 0.

    # If only one lat/lon value then set up the observer location
    # for Astral.  If more than one, it will get set up in the loop
    if lat.size == 1 and ASTRAL:
        loc = Observer(latitude=lat, longitude=lon)

    # Loop through each time to ensure that the sunrise/set calcuations
    # are correct for each time and lat/lon if multiple
    results = []
    time = obj['time'].values
    for i in range(len(time)):
        # Set up an observer if multiple lat/lon
        if lat.size > 1:
            if ASTRAL:
                loc = Observer(latitude=lat[i], longitude=lon[i])
            else:
                s = a.sun_utc(pd.to_datetime(time[i]), lat[i], lon[i])
        elif ASTRAL is False:
            s = a.sun_utc(pd.to_datetime(time[i]), float(lat), float(lon))

        # Get sunrise and sunset
        if ASTRAL:
            sr = sun.sunrise(loc, pd.to_datetime(time[i]))
            ss = sun.sunset(loc, pd.to_datetime(time[i]))
        else:
            sr = s['sunrise']
            ss = s['sunset']

        # Set longname
        longname = 'Daylight indicator; 0-Night; 1-Sun'

        # Check to see if dawn/dusk calculations can be performed before preceeding
        if dawn_dusk:
            try:
                if ASTRAL:
                    dwn = sun.dawn(loc, pd.to_datetime(time[i]))
                    dsk = sun.dusk(loc, pd.to_datetime(time[i]))
                else:
                    if lat.size > 1:
                        dsk = a.dusk_utc(pd.to_datetime(time[i]), lat[i], lon[i])
                        dwn = a.dawn_utc(pd.to_datetime(time[i]), lat[i], lon[i])
                    else:
                        dsk = a.dusk_utc(pd.to_datetime(time[i]), float(lat), float(lon))
                        dwn = a.dawn_utc(pd.to_datetime(time[i]), float(lat), float(lon))
            except ValueError:
                print('Dawn/Dusk calculations are not available at this location')
                dawn_dusk = False

        if dawn_dusk and ASTRAL:
            # If dawn_dusk is True, add 2 more indicators
            longname += '; 2-Dawn; 3-Dusk'
            # Need to ensure if the sunset if off a day to grab the previous
            # days value to catch the early UTC times
            if ss.day > sr.day:
                if ASTRAL:
                    ss = sun.sunset(loc, pd.to_datetime(time[i] - np.timedelta64(1, 'D')))
                    dsk = sun.dusk(loc, pd.to_datetime(time[i] - np.timedelta64(1, 'D')))
                else:
                    if lat.size > 1:
                        dsk = a.dusk_utc(pd.to_datetime(time[i]) - np.timedelta64(1, 'D'),
                                         lat[i], lon[i])
                        s = a.sun_utc(pd.to_datetime(time[i]) - np.timedelta64(1, 'D'),
                                      lat[i], lon[i])
                    else:
                        dsk = a.dusk_utc(pd.to_datetime(time[i]) - np.timedelta64(1, 'D'),
                                         float(lat), float(lon))
                        s = a.sun_utc(pd.to_datetime(time[i]) - np.timedelta64(1, 'D'),
                                      float(lat), float(lon))
                    ss = s['sunset']

                if dwn <= pd.to_datetime(time[i], utc=True) < sr:
                    results.append(2)
                elif ss <= pd.to_datetime(time[i], utc=True) < dsk:
                    results.append(3)
                elif not(dsk <= pd.to_datetime(time[i], utc=True) < dwn):
                    results.append(1)
                else:
                    results.append(0)
            else:
                if dwn <= pd.to_datetime(time[i], utc=True) < sr:
                    results.append(2)
                elif sr <= pd.to_datetime(time[i], utc=True) < ss:
                    results.append(1)
                elif ss <= pd.to_datetime(time[i], utc=True) < dsk:
                    results.append(3)
                else:
                    results.append(0)
        else:
            if ss.day > sr.day:
                if ASTRAL:
                    ss = sun.sunset(loc, pd.to_datetime(time[i] - np.timedelta64(1, 'D')))
                else:
                    s = a.sun_utc(pd.to_datetime(time[i]) - np.timedelta64(1, 'D'), lat, lon)
                    ss = s['sunset']
                results.append(int(not(ss < pd.to_datetime(time[i], utc=True) < sr)))
            else:
                results.append(int(sr < pd.to_datetime(time[i], utc=True) < ss))

    # Add results to object and return
    obj['sun_variable'] = ('time', np.array(results),
                           {'long_name': longname, 'units': ' '})

    return obj


def get_solar_azimuth_elevation(latitude=None, longitude=None, time=None, library='skyfield',
                                temperature_C='standard', pressure_mbar='standard'):
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

    result = {'elevation': None, 'azimuth': None, 'distance': None}

    if library == 'skyfield':
        planets = load(skyfield_bsp_file)
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
        location = earth + wgs84.latlon(latitude * N, longitude * W)
        astrometric = location.at(t).observe(sun)
        alt, az, distance = astrometric.apparent().altaz(temperature_C=temperature_C,
                                                         pressure_mbar=pressure_mbar)
        result = (alt.degrees, az.degrees, distance.au)

    return result


def get_sunrise_sunset_noon(latitude=None, longitude=None, date=None, library='skyfield',
                            timezone=False):
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
        eph = load(skyfield_bsp_file)
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
            t0 = min(noon) - timedelta(days=90)
            t1 = max(noon) + timedelta(days=90)
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
            sunrise = temp_sunrise[sunrise_index: sunset_index]
            sunset = temp_sunset[sunrise_index: sunset_index]

    if timezone is False:
        for ii in range(0, sunset.size):
            sunrise[ii] = sunrise[ii].replace(tzinfo=None)
            sunset[ii] = sunset[ii].replace(tzinfo=None)
        for ii in range(0, noon.size):
            noon[ii] = noon[ii].replace(tzinfo=None)

    return sunrise, sunset, noon


def is_sun_visible(latitude=None, longitude=None, date_time=None):
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
        raise ValueError('The date_time values entered into is_sun_visible() '
                         'do not match input types.')

    ts = load.timescale()
    eph = load(skyfield_bsp_file)

    t0 = ts.from_datetimes(sf_dates)
    location = wgs84.latlon(latitude, longitude)
    f = almanac.sunrise_sunset(eph, location)
    sun_up = f(t0)

    return sun_up
