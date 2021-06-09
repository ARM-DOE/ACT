"""
This module contains functions for correcting ceilometer data

"""
import numpy as np


def correct_ceil(obj, fill_value=1e-7, var_name='backscatter'):
    """
    This procedure corrects ceilometer data by filling all zero and
    negative values of backscatter with fill_value and then converting
    the backscatter data into logarithmic space.

    Parameters
    ----------
    obj : Dataset object
        The ceilometer dataset to correct. The backscatter data should be
        in linear space.
    var_name : str
        The variable name of data in the object.
    fill_value : float
        The fill_value to use. The fill_value is entered in linear space.

    Returns
    -------
    obj : Dataset object
        The ceilometer dataset containing the corrected values.

    """
    data = obj[var_name].data
    data[data <= 0] = fill_value
    data = np.log10(data)

    obj[var_name].values = data

    return obj
