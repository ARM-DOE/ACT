"""
Functions for calculated cloud base height that are instrument agnostic.

"""

import numpy as np
import xarray as xr
from scipy import ndimage


def generic_sobel_cbh(
    ds,
    variable=None,
    height_dim=None,
    var_thresh=None,
    fill_na=None,
    return_thresh=False,
    filter_type='uniform',
    edge_thresh=5.0,
):
    """
    Function for calculating cloud base height from lidar/radar data
    using a basic sobel filter and thresholding. Note, this was not
    initially based on any published work, but a lit review indicates
    that there have been similar methods employed to detect boundary
    layer heights.

    NOTE: The returned variable now appends the field name of the
    data used to generate the CBH as part of the variable name.  cbh_sobel_[varname]

    Parameters
    ----------
    ds : ACT xarray.Dataset
        ACT xarray dataset where data are stored.
    variable : string
        Variable on which to process.
    height_dim : string
        Height variable to use for CBH values.
    var_thresh : float
        Thresholding for variable if needed.
    fill_na : float
        Value to fill nans with in DataArray if any.
    filter_type : string
        Currently the only option is for uniform filtering.
        uniform: Apply uniform filtering after the sobel filter?  Applies a standard area of 3x3 filtering
        None: Excludes the filtering
    edge_thresh : float
        Threshold value for finding the edge after the sobel filtering.
        If the signal is not strong, this may need to be lowered

    Returns
    -------
    new_ds : ACT xarray.Dataset
        ACT xarray dataset with cbh values included as variable.

    Examples
    --------
    In testing on the ARM KAZR and MPL data, the following methods
    tended to work best for thresholding/corrections/etc.

    .. code-block:: python

        kazr = act.retrievals.cbh.generic_sobel_cbh(
            kazr, variable="reflectivity_copol", height_dim="range", var_thresh=-10.0
        )

        mpl = act.corrections.mpl.correct_mpl(mpl)
        mpl.range_bins.values = mpl.height.values[0, :] * 1000.0
        mpl.range_bins.attrs["units"] = "m"
        mpl["signal_return_co_pol"].values[:, 0:10] = 0.0
        mpl = act.retrievals.cbh.generic_sobel_cbh(
            mpl,
            variable="signal_return_co_pol",
            height_dim="range_bins",
            var_thresh=10.0,
            fill_na=0.0,
        )

        ceil = act.retrievals.cbh.generic_sobel_cbh(
            ceil,
            variable="backscatter",
            height_dim="range",
            var_thresh=1000.0,
            fill_na=0,
        )

    """
    if variable is None:
        return
    if fill_na is None:
        fill_na = var_thresh

    # Pull data into Standalone DataArray
    da = ds[variable]

    # Apply thresholds if set
    if var_thresh is not None:
        da = da.where(da.values > var_thresh)

    # Fill with fill_na values
    da = da.fillna(fill_na)

    # If return_thresh is True, replace variable data with
    # thresholded data
    if return_thresh is True:
        ds[variable].values = da.values

    # Apply Sobel filter to data and smooth the results
    data = da.values.tolist()
    edge = ndimage.sobel(data)
    if filter_type == 'uniform':
        edge = ndimage.uniform_filter(edge, size=3, mode='nearest')

    # Create Data Array
    edge_da = xr.DataArray(edge, dims=ds[variable].dims)

    # Filter some of the resulting edge data to get defined edges
    edge_da = edge_da.where(edge_da > edge_thresh)
    edge_da = edge_da.fillna(fill_na)

    # Do a diff along the height dimension to define edge
    diff = edge_da.diff(dim=1).values

    # Get height variable to use for cbh
    height = ds[height_dim].values

    # Run through times and find the height
    cbh = []
    for i in range(np.shape(diff)[0]):
        try:
            index = np.where(diff[i, :] > edge_thresh)[0]
        except ValueError():
            index = []
        if len(np.shape(height)) > 1:
            ht = height[i, :]
        else:
            ht = height

        if len(index) > 0:
            cbh.append(ht[index[0] + 1])
        else:
            cbh.append(np.nan)

    # Create DataArray to add to the dataset
    var_name = 'cbh_sobel_' + variable
    da = xr.DataArray(cbh, dims=['time'], coords=[ds['time'].values])
    ds[var_name] = da
    ds[var_name].attrs['long_name'] = ' '.join(
        ['CBH calculated from', variable, 'using sobel filter']
    )
    ds[var_name].attrs['units'] = ds[height_dim].attrs['units']

    return ds
