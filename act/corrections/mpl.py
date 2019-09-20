import numpy as np
import xarray as xr
import warnings


def correct_mpl(obj):
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

    Note: Deadtime and darkcount corrections are not being applied yet.

    Parameters
    ----------
    obj : Dataset object
        The ACT object.

    Returns
    -------
    obj : Dataset object
        The ACT Object containing the corrected values.

    """
    # Get some variables before processing begins
    act = obj.act

    # Overlap Correction Variable
    op = obj['overlap_correction'].values[0, :]
    op_height = obj['overlap_correction_heights'].values[0, :]

    # 1 - Remove negative height data
    obj = obj.where(obj.height > 0., drop=True)
    height = obj['height'].values

    # If act is not in object, add back in.  Fixed in xarray v0.13.0
    if hasattr(obj, 'act') is False:
        obj.act = act

    # Get indices for calculating background
    var_names = ['signal_return_co_pol', 'signal_return_cross_pol']
    ind = [obj.height.shape[1] - 50, obj.height.shape[1] - 2]

    # Subset last gates into new dataset
    dummy = obj.isel(range_bins=xr.DataArray(np.arange(ind[0], ind[1])))

    # Turn off warnings
    warnings.filterwarnings("ignore")

    # Run through co and cross pol data for corrections
    co_bg = dummy[var_names[0]]
    co_bg = co_bg.where(co_bg > -9998.)
    co_bg = co_bg.mean(dim='dim_0').values

    x_bg = dummy[var_names[1]]
    x_bg = x_bg.where(x_bg > -9998.)
    x_bg = x_bg.mean(dim='dim_0').values

    # Seems to be the fastest way of removing background signal at the moment
    co_data = obj[var_names[0]].where(obj[var_names[0]] > 0).values
    x_data = obj[var_names[1]].where(obj[var_names[1]] > 0).values
    for i in range(len(obj['time'].values)):
        co_data[i, :] = co_data[i, :] - co_bg[i]
        x_data[i, :] = x_data[i, :] - x_bg[i]

    # After Pulse Correction Variable
    co_ap = obj['afterpulse_correction_co_pol'].values
    x_ap = obj['afterpulse_correction_cross_pol'].values

    for j in range(len(obj['range_bins'].values)):
        # Afterpulse Correction
        co_data[:, j] = co_data[:, j] - co_ap[:, j]
        x_data[:, j] = x_data[:, j] - x_ap[:, j]

        # R-Squared Correction
        co_data[:, j] = co_data[:, j] * height[:, j] ** 2.
        x_data[:, j] = x_data[:, j] * height[:, j] ** 2.

        # Overlap Correction
        idx = (np.abs(op_height - height[0, j])).argmin()
        co_data[:, j] = co_data[:, j] * op[idx]
        x_data[:, j] = x_data[:, j] * op[idx]

    # Create the co/cross ratio variable
    ratio = (x_data / co_data) * 100.
    obj['cross_co_ratio'] = obj[var_names[0]].copy(data=ratio)

    # Convert data to decibels
    co_data = 10. * np.log10(co_data)
    x_data = 10. * np.log10(x_data)

    # Write data to object
    obj[var_names[0]].values = co_data
    obj[var_names[1]].values = x_data

    return obj
