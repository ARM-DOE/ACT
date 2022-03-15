"""
This module contains corrections for micropulse lidars

"""
import warnings

import numpy as np
import xarray as xr


def correct_mpl(
    obj,
    co_pol_var_name='signal_return_co_pol',
    cross_pol_var_name='signal_return_cross_pol',
    co_pol_afterpuls_var_name='afterpulse_correction_co_pol',
    cross_pol_afterpulse_var_name='afterpulse_correction_cross_pol',
    overlap_corr_var_name='overlap_correction',
    overlap_corr_heights_var_name='overlap_correction_heights',
    range_bins_var_name='range_bins',
    height_var_name='height',
    ratio_var_name='cross_co_ratio',
):
    """
    This procedure corrects MPL data:
    1.) Throw out data before laser firing (heights < 0).
    2.) Remove background signal.
    3.) Afterpulse Correction - Subtraction of (afterpulse-darkcount).
    NOTE: Currently the Darkcount in VAPS is being calculated as
    the afterpulse at ~30km. But that might not be absolutely
    correct and we will likely start providing darkcount profiles
    ourselves along with other corrections.
    4.) Range Correction.
    5.) Overlap Correction (Multiply).

    If the height variable changes between netCDF files, Xarray will turn the height
    dimention variables from a 1D to a 2D array. This will cause issues with processing
    and other data manipulation. To fix this the 2D height will be converted to a 1D
    array by using the median value for each height value.

    Note: Deadtime and darkcount corrections are not being applied yet.

    Parameters
    ----------
    obj : Xarray Dataset
        The ACT Xarray Dataset containg data
    co_pol_var_name : str
        The Co-polar variable name used in Dataset
    cross_pol_var_name : str
        The Cross-polar variable name used in Dataset
    co_pol_afterpuls_var_name : str
        The Co-polar after pluse variable name used in Dataset
    cross_pol_afterpulse_var_name : str
        The Cross-polar afterpulse variable name used in Dataset
    overlap_corr_var_name : str
        The overlap correction variable name used in Dataset
    overlap_corr_heights_var_name : str
        The overlap correction height variable name used in Dataset
    range_bins_var_name : str
        The range bins variable name used in Dataset
    height_var_name : str
        The height variable name used in Dataset
    ratio_var_name : str
        The variable name to use for newly created variable of co/cross ratio.
        Set to None if do not want this new variables created in Dataset.

    Returns
    -------
    obj : Xarray Dataset
        The ACT Object containing the corrected values. The orginal Xarray Dataset
        passed in is modified.

    """

    data_dims = obj[co_pol_var_name].dims

    # Overlap Correction Variable
    op = obj[overlap_corr_var_name].values
    if len(op.shape) > 1:
        op = op[0, :]

    op_height = obj[overlap_corr_heights_var_name].values
    if len(op_height.shape) > 1:
        op_height = op_height[0, :]

    # Check if height has dimentionality of time and height. If so reduce height
    # to only dimentionality of height in object before removing values less than 0.
    if len(obj[height_var_name].shape) > 1:
        reduce_dim_name = {'time'} & set(obj[height_var_name].dims)
        obj[height_var_name] = obj[height_var_name].reduce(
            func=np.median, dim=reduce_dim_name, keep_attrs=True
        )

    # 1 - Remove negative height data
    obj = obj.where(obj[height_var_name] > 0.0, drop=True)
    height = obj[height_var_name].values

    # Get indices for calculating background
    if len(obj.height.shape) > 1:
        ind = [obj.height.shape[1] - 50, obj.height.shape[1] - 2]
    else:
        ind = [obj.height.shape[0] - 50, obj.height.shape[0] - 2]

    # Subset last gates into new dataset
    dummy = obj.isel(range_bins=xr.DataArray(np.arange(ind[0], ind[1])))

    # Turn off warnings
    warnings.filterwarnings('ignore')

    # Run through co and cross pol data for corrections
    co_bg = dummy[co_pol_var_name]
    co_bg = co_bg.where(co_bg > -9998.0)
    co_bg = co_bg.mean(dim='dim_0').values

    x_bg = dummy[cross_pol_var_name]
    x_bg = x_bg.where(x_bg > -9998.0)
    x_bg = x_bg.mean(dim='dim_0').values

    # Seems to be the fastest way of removing background signal at the moment
    co_data = obj[co_pol_var_name].where(obj[co_pol_var_name] > 0).values
    x_data = obj[cross_pol_var_name].where(obj[cross_pol_var_name] > 0).values
    for i in range(len(obj['time'].values)):
        co_data[i, :] = co_data[i, :] - co_bg[i]
        x_data[i, :] = x_data[i, :] - x_bg[i]

    # After Pulse Correction Variable
    co_ap = obj[co_pol_afterpuls_var_name].values
    # Fix dimentionality if backwards
    co_ap_dims = obj[co_pol_afterpuls_var_name].dims
    if len(co_ap_dims) > 1 and co_ap_dims[::-1] == data_dims:
        co_ap = np.transpose(co_ap)

    x_ap = obj[cross_pol_afterpulse_var_name].values

    # Fix dimentionality if backwards
    x_ap_dims = obj[cross_pol_afterpulse_var_name].dims
    if len(x_ap_dims) > 1 and x_ap_dims[::-1] == data_dims:
        x_ap = np.transpose(x_ap)

    # Afterpulse Correction
    co_data = co_data - co_ap
    x_data = x_data - x_ap

    # R-Squared Correction
    co_data = co_data * height**2
    x_data = x_data * height**2

    # Overlap Correction
    for j in range(obj[range_bins_var_name].size):
        if len(height.shape) > 1:
            idx = (np.abs(op_height - height[0, j])).argmin()
        else:
            # Overlap Correction
            idx = (np.abs(op_height - height[j])).argmin()

        co_data[:, j] = co_data[:, j] * op[idx]
        x_data[:, j] = x_data[:, j] * op[idx]

    # Create the co/cross ratio variable
    if ratio_var_name is not None:
        ratio = (x_data / (x_data + co_data)) * 100.0
        obj[ratio_var_name] = obj[co_pol_var_name].copy(data=ratio)
        obj[ratio_var_name].attrs['long_name'] = 'Cross-pol / Co-pol ratio * 100'
        obj[ratio_var_name].attrs['units'] = 'LDR'
        try:
            del obj[ratio_var_name].attrs['ancillary_variables']
            del obj[ratio_var_name].attrs['description']
        except KeyError:
            pass

    # Convert data to decibels
    co_data = 10.0 * np.log10(co_data)
    x_data = 10.0 * np.log10(x_data)

    # Write data to object
    obj[co_pol_var_name].values = co_data
    obj[cross_pol_var_name].values = x_data

    # Update units
    obj[co_pol_var_name].attrs['units'] = f"10 * log10({obj[co_pol_var_name].attrs['units']})"
    obj[cross_pol_var_name].attrs['units'] = f"10 * log10({obj[cross_pol_var_name].attrs['units']})"

    return obj
