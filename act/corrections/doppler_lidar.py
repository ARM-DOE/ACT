import numpy as np


def correct_dl(obj, var_name='attenuated_backscatter', fill_value=1e-7):
    """
    This procedure corrects doppler lidar data by filling all zero and
    negative values of backscatter with fill_value and then converting
    the backscatter data into logarithmic space.

    Parameters
    ----------
    obj : Dataset object
        The doppler lidar dataset to correct. The backscatter data should be
        in linear space.
    var_name : str
        The variable name of data in the object.
    fill_value : float
        The fill_value to use. The fill_value is entered in linear space.

    Returns
    -------
    obj : Dataset object
        The doppler lidar dataset containing the corrected values.

    """
    backscat = obj[var_name].values

    # This will get the name of the coordinate dimension so it's not assumed
    # via position or name
    height_name = list(set(obj[var_name].dims) - set(['time']))[0]
    height = obj[height_name].values
    height = height ** 2

    # Doing this trick with height to change the array shape so it
    # will broadcast correclty against backscat
    backscat = backscat * height[None, :]
    backscat[backscat <= 0] = fill_value
    obj[var_name].values = np.log10(backscat)

    # Updating the units to correctly indicate the values are log values
    obj[var_name].attrs['units'] = 'log(' + obj[var_name].attrs['units'] + ')'

    return obj
