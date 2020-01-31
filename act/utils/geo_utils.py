"""
act.utils.geo_utils
--------------------

Module containing utilities for geographic calculations

"""

import numpy as np


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
