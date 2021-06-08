"""
This module contains functions for correcting data for ship motion

"""
import numpy as np


def correct_wind(obj, wspd_name='wind_speed', wdir_name='wind_direction',
                 heading_name='yaw', cog_name='course_over_ground',
                 sog_name='speed_over_ground'):
    """
    This procedure corrects wind speed and direction for ship motion
    based on equations from NOAA tech. memo. PSD-311. A Guide to Making
    Climate Quality Meteorological and Flux Measurements at Sea.
    https://www.go-ship.org/Manual/fluxhandbook_NOAA-TECH%20PSD-311v3.pdf

    Parameters
    ----------
    obj : Dataset object
        The ceilometer dataset to correct. The backscatter data should be
        in linear space.
    wspd_name : string
        Wind speed variable name.
    wdir_name : string
        Wind direction variable name.
    heading_name : string
        Navigiation heading variable name.
    cog_name : string
        Course over ground variable name.
    sog_name : string
        Speed over ground variable name.

    Returns
    -------
    obj : Dataset object
        The dataset containing the corrected values.

    References
    ----------
    Bradley, F. and Farall. C. (2007) A Guide to Making Climate Quality Meteorological
    and Flux Measurements at Sea. Boulder, CO, NOAA, Earth System Research Laboratory,
    Physical Sciences Division, 44pp. & appendices. (NOAA Technical Memorandum OAR PSD-311).
    http://hdl.handle.net/11329/386

    """
    # Set variables to be used and convert to radians
    rels = obj[wspd_name]
    reld = np.deg2rad(obj[wdir_name])
    head = np.deg2rad(obj[heading_name])
    cog = np.deg2rad(obj[cog_name])
    sog = obj[sog_name]

    # Calculate winds based on method in the document denoted above
    relsn = rels * np.cos(head + reld)
    relse = rels * np.sin(head + reld)

    sogn = sog * np.cos(cog)
    soge = sog * np.sin(cog)

    un = relsn - sogn
    ue = relse - soge

    dirt = np.mod(np.rad2deg(np.arctan2(ue, un)) + 360., 360)
    ut = np.sqrt(un ** 2. + ue ** 2)

    # Create data arrays and add corrected wind direction and speed
    # to the initial object that was passed in
    wdir_da = obj[wdir_name].copy(data=dirt)
    wdir_da.attrs['long_name'] = 'Wind direction corrected to ship motion'
    obj[wdir_name + '_corrected'] = wdir_da

    wspd_da = obj[wspd_name].copy(data=ut)
    wspd_da.attrs['long_name'] = 'Wind speed corrected to ship motion'
    obj[wspd_name + '_corrected'] = wspd_da

    return obj
