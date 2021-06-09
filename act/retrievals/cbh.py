"""
Functions for calculated cloud base height that are instrument agnostic.

"""

import numpy as np
import xarray as xr
from scipy import ndimage


def generic_sobel_cbh(obj, variable=None, height_dim=None,
                      var_thresh=None, fill_na=None,
                      return_thresh=False):
    """
    Function for calculating cloud base height from lidar/radar data
    using a basic sobel filter and thresholding. Note, this was not
    initially based on any published work, but a lit review indicates
    that there have been similar methods employed to detect boundary
    layer heights.

    Parameters
    ----------
    obj : ACT Object
        ACT object where data are stored.
    variable : string
        Variable on which to process.
    height_dim : string
        Height variable to use for CBH values.
    var_thresh : float
        Thresholding for variable if needed.
    fill_na : float
        What to fill nans with in DataArray if any.

    Returns
    -------
    new_obj : ACT Object
        ACT Object with cbh values included as variable.

    Examples
    --------
    In testing on the ARM KAZR and MPL data, the following methods
    tended to work best for thresholding/corrections/etc.

    .. code-block:: python

        kazr = act.retrievals.cbh.generic_sobel_cbh(kazr,variable='reflectivity_copol',
                                                    height_dim='range', var_thresh=-10.)

        mpl = act.corrections.mpl.correct_mpl(mpl)
        mpl.range_bins.values = mpl.height.values[0,:]*1000.
        mpl.range_bins.attrs['units'] = 'm'
        mpl['signal_return_co_pol'].values[:,0:10] = 0.
        mpl = act.retrievals.cbh.generic_sobel_cbh(mpl,variable='signal_return_co_pol',
                                            height_dim='range_bins',var_thresh=10.,
                                            fill_na=0.)

        ceil = act.retrievals.cbh.generic_sobel_cbh(ceil,variable='backscatter',
                                            height_dim='range', var_thresh=1000.,
                                            fill_na=0)

    """
    if variable is None:
        return
    if fill_na is None:
        fill_na = var_thresh

    # Pull data into Standalone DataArray
    data = obj[variable]

    # Apply thresholds if set
    if var_thresh is not None:
        data = data.where(data.values > var_thresh)

    # Fill with fill_na values
    data = data.fillna(fill_na)

    # If return_thresh is True, replace variable data with
    # thresholded data
    if return_thresh is True:
        obj[variable].values = data.values

    # Apply Sobel filter to data and smooth the results
    data = data.values
    edge = ndimage.sobel(data)
    edge = ndimage.uniform_filter(edge, size=3, mode='nearest')

    # Create Data Array
    edge_obj = xr.DataArray(edge, dims=obj[variable].dims)

    # Filter some of the resulting edge data to get defined edges
    edge_obj = edge_obj.where(edge_obj > 5.)
    edge_obj = edge_obj.fillna(fill_na)

    # Do a diff along the height dimension to define edge
    diff = edge_obj.diff(dim=1)

    # Get height variable to use for cbh
    height = obj[height_dim].values

    # Run through times and find the height
    cbh = []
    for i in range(np.shape(diff)[0]):
        index = np.where(diff[i, :] > 5.)[0]
        if len(np.shape(height)) > 1:
            ht = height[i, :]
        else:
            ht = height

        if len(index) > 0:
            cbh.append(ht[index[0] + 1])
        else:
            cbh.append(np.nan)

    # Create DataArray to add to Object
    da = xr.DataArray(cbh, dims=['time'], coords=[obj['time'].values])
    obj['cbh_sobel'] = da
    obj['cbh_sobel'].attrs['long_name'] = ' '.join(['CBH calculated from',
                                                    variable, 'using sobel filter'])
    obj['cbh_sobel'].attrs['units'] = obj[height_dim].attrs['units']

    return obj
