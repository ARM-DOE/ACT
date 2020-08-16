"""
act.utils.geo_utils
--------------------

Module containing utilities for geographic calculations,
including solar calculations

"""

import numpy as np
import pandas as pd
import astral
try:
    from astral.sun import sunrise, sunset, dusk, dawn
    from astral import Observer
    ASTRAL = True
except ImportError:
    ASTRAL = False


def destination_azimuth_distance(lat, lon, az, dist):
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
        Distance in meters.

    Returns
    -------
    lat2 : float
        Latitude of new point.
    lon2 : float
        Longitude of new point.

    """
    # Volumetric Mean Radius of Earth
    R = 6371.

    # Convert az to radian
    brng = np.radians(az)

    # Assuming meters as input
    d = dist / 1000.

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
    obj : act object
        ACT object
    latitude : str
        Latitude variable, default will look for matching variables
        in object
    longitude : str
        Longitude variable, default will look for matching variables
        in object
    solar_angle : float
        Number of degress to use for dawn/dusk calculations
    dawn_dusk : boolean
         If set to True, will add values 2 (dawn) and 3 (dusk) to the solar variable

    Returns
    -------
    obj : act object
        ACT object
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
            sr = sunrise(loc, pd.to_datetime(time[i]))
            ss = sunset(loc, pd.to_datetime(time[i]))
        else:
            sr = s['sunrise']
            ss = s['sunset']

        # Set longname
        longname = 'Daylight indicator; 0-Night; 1-Sun'

        # Check to see if dawn/dusk calculations can be performed before preceeding
        if dawn_dusk:
            try:
                if ASTRAL:
                    dwn = dawn(loc, pd.to_datetime(time[i]))
                    dsk = dusk(loc, pd.to_datetime(time[i]))
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
                    ss = sunset(loc, pd.to_datetime(time[i] - np.timedelta64(1, 'D')))
                    dsk = dusk(loc, pd.to_datetime(time[i] - np.timedelta64(1, 'D')))
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
                    ss = sunset(loc, pd.to_datetime(time[i] - np.timedelta64(1, 'D')))
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
