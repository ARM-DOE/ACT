"""
This module contains functions for correcting raman lidar data

"""
import numpy as np


def correct_rl(obj, var_name='depolarization_counts_high', fill_value=1e-7,
               range_normalize_log_values=False):
    """
    This procedure corrects raman lidar data by filling all zero and
    negative values of backscatter with fill_value and then converting
    the backscatter data into logarithmic space. It will also look for
    a coordinate height dimension, and if one is not found will create
    using values from global attributes.

    Parameters
    ----------
    obj : Dataset object
        The doppler lidar dataset to correct. The backscatter data should be
        in linear space.
    var_name : str
        The variable name of data in the object.
    fill_value : float
        The fill_value to use. The fill_value is entered in linear space.
    range_normalize_log_values : boolean
        Option to range normalize and convert to log scale of counts values.

    Returns
    -------
    obj : Dataset object
        The raman lidar dataset containing the corrected values.

    """
    # This will get the name of the coordinate dimension so it's not assumed
    # via position or name.
    height_name = list(set(obj[var_name].dims) - set(['time']))[0]

    # Check if the height dimension is a variable in the object. If not
    # use global attributes to derive the values and put into object.
    #
    # Asking for a variable name in the object that is also a dimension
    # but does not exist will return an index array starting at 0.
    height = obj[height_name].values
    if height_name not in list(obj.data_vars):
        # Determin which mode we are correcting
        level = height_name.split('_')[0]
        att_name = [i for i in list(obj.attrs) if
                    'vertical_resolution_' + level + '_channels' in i]

        # Extract information from global attributes
        bin_size_raw = (obj.attrs[att_name[0]]).split()
        bins_before_shot = float(obj.attrs['number_of_bins_before_shot'])
        bin_size = float(bin_size_raw[0])
        height = (height + 1) * bin_size
        height = height - bins_before_shot * bin_size
        obj[height_name] = (height_name, height,
                            {'long_name': 'Height above ground',
                             'units': bin_size_raw[1]})

    if range_normalize_log_values:
        height = height ** 2  # Range normalize values
        backscat = obj[var_name].values
        # Doing this trick with height to change the array shape so it
        # will broadcast correclty against backscat
        backscat = backscat * height[None, :]
        backscat[backscat <= 0] = fill_value
        if np.shape(obj[var_name].values) != np.shape(np.log10(backscat)):
            obj[var_name].values = np.reshape(np.log10(backscat),
                                              np.shape(obj[var_name].values))
        else:
            obj[var_name].values = np.log10(backscat)

        # Updating the units to correctly indicate the values are log values
        obj[var_name].attrs['units'] = 'log(' + obj[var_name].attrs['units'] + ')'

    return obj
