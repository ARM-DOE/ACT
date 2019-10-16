"""
act.utils.ship_utils
--------------------

Module containing utilities for ship data

"""
import pyproj
import dask
import xarray as xr
import numpy as np


def calc_cog_sog(obj):
    """
    This procedure will create a new ACT dataset whose coordinates are
    designated to be the variables in a given list. This helps make data
    slicing via xarray and visualization easier.

    Parameters
    ----------
    obj: ACT Dataset
        The ACT Dataset to modify the coordinates of.

    Returns
    -------
    obj: ACT Dataset
        Returns object with course_over_ground and speed_over_ground variables

    """

    # Convert data to 1 minute in order to get proper values
    new_obj = obj.resample(time='1min').nearest()

    # Get lat and lon data
    if 'lat' in new_obj:
        lat = new_obj['lat']
    elif 'latitude' in new_obj:
        lat = new_obj['latitude']
    else:
        return new_obj

    if 'lon' in new_obj:
        lon = new_obj['lon']
    elif 'longitude' in new_obj:
        lat = new_obj['longitude']
    else:
        return new_obj

    # Set pyproj Geod
    _GEOD = pyproj.Geod(ellps='WGS84')

    # Set up delayed tasks for dask
    task = []
    time = new_obj['time'].values
    for i in range(len(lat) - 1):
        task.append(dask.delayed(proc_scog)
                    (_GEOD, lon[i + 1], lat[i + 1], lon[i], lat[i],
                    time[i], time[i + 1]))

    # Compute and process results Adding 2 values
    # to the end to make up for the missing times
    results = dask.compute(*task)
    sog = [r[0] for r in results]
    sog.append(sog[-1])
    sog.append(sog[-1])
    cog = [r[1] for r in results]
    cog.append(cog[-1])
    cog.append(cog[-1])
    time = np.append(time, time[-1] + np.timedelta64(1, 'm'))

    atts = {'long_name': 'Speed over ground', 'units': 'm/s'}
    sog_da = xr.DataArray(sog, coords={'time': time}, dims=['time'], attrs=atts)
    sog_da = sog_da.resample(time='1s').nearest()

    atts = {'long_name': 'Course over ground', 'units': 'deg'}
    cog_da = xr.DataArray(cog, coords={'time': time}, dims=['time'], attrs=atts)
    cog_da = cog_da.resample(time='1s').nearest()

    obj['course_over_ground'] = cog_da
    obj['speed_over_ground'] = sog_da

    return obj


def proc_scog(_GEOD, lon2, lat2, lon1, lat1, time1, time2):
    """
    This procedure is to only be used by the calc_cog_sog
    function for dask delayed processing

    """
    cog, baz, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    tdiff = (time2 - time1) / np.timedelta64(1, 's')
    sog = dist / tdiff
    if cog < 0:
        cog = 360. + cog
    if sog < 0.5:
        cog = np.nan

    return sog, cog, dist
