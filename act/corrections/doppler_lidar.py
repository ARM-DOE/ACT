"""
This module contains functions for correcting doppler lidar data

"""
import numpy as np


def correct_dl(obj, var_name='attenuated_backscatter', range_normalize=True,
               fill_value=1e-7):
    """
    This procedure corrects doppler lidar data by filling all zero and
    negative values of backscatter with fill_value and then converting
    the backscatter data into logarithmic space. Will also range normalize
    the values by multipling by the range^2 with default units.

    Parameters
    ----------
    obj : Dataset object
        The doppler lidar dataset to correct. The backscatter data should be
        in linear space.
    var_name : str
        The variable name of data in the object.
    range_normalize : bool
        Option to range normalize the data.
    fill_value : float
        The fill_value to use. The fill_value is entered in linear space.

    Returns
    -------
    obj : Dataset object
        The doppler lidar dataset containing the corrected values.

    """
    data = obj[var_name].values

    if range_normalize:
        # This will get the name of the coordinate dimension so it's not assumed
        # via position or name
        height_name = list(set(obj[var_name].dims) - set(['time']))[0]
        height = obj[height_name].values
        height = height / np.max(height)
        data = data * height ** 2

    data[data <= 0] = fill_value
    obj[var_name].values = np.log10(data)

    # Updating the units to correctly indicate the values are log values
    if range_normalize:
        obj[var_name].attrs['units'] = 'log( Range normalized ' + obj[var_name].attrs['units'] + ')'
    else:
        obj[var_name].attrs['units'] = 'log(' + obj[var_name].attrs['units'] + ')'

    return obj
