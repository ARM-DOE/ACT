"""
This module contains functions for correcting ceilometer data

"""
import numpy as np


def correct_ceil(ds, fill_value=1e-7, var_name='backscatter'):
    """
    This procedure corrects ceilometer data by filling all zero and
    negative values of backscatter with fill_value and then converting
    the backscatter data into logarithmic space.

    Parameters
    ----------
    ds : xarray.Dataset
        The ceilometer dataset to correct. The backscatter data should be
        in linear space.
    var_name : str
        The variable name of data in the dataset.
    fill_value : float
        The fill_value to use. The fill_value is entered in linear space.

    Returns
    -------
    ds : xarray.Dataset
        The ceilometer dataset containing the corrected values.

    """
    data = ds[var_name].data
    data[data <= 0] = fill_value
    data = np.log10(data)

    ds[var_name].values = data
    if 'units' in ds[var_name].attrs:
        ds[var_name].attrs['units'] = 'log(' + ds[var_name].attrs['units'] + ')'
    else:
        ds[var_name].attrs['units'] = 'log(unknown)'

    return ds
