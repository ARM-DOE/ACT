"""
Functions for aeri retrievals.

"""

import numpy as np
from scipy.optimize import brentq
from act.retrievals.irt import irt_response_function, sum_function_irt


def aeri2irt(aeri_ds, wnum_name='wnum', rad_name='mean_rad', hatch_name='hatchOpen',
             tolerance=0.1, temp_low=150.0, temp_high=320.0, maxiter=200):
    """
    This function will integrate over the correct wavenumber values to produce
    the effective IRT temperature.

    As a note from the ARM IRT Instrument Handbook
    A positive bias of the sky temperature is exhibited by the downwelling IRT,
    compared to the AERI, during clear-sky conditions when the sky temperature
    is less than ~180K. The effect depends on the characteristics of the
    individual IRT and the internal reference temperature of the IRT. The
    greatest difference compared to AERI will occur when the sky is very clear,
    dry, and cold and the ambient temperature is relatively hot, maximizing the
    difference in temperature between the sky and instrument, and the
    calibration of the IRT at the lower limit of 223K was not performed
    accurately. This bias is especially apparent at high-latitude sites
    (e.g., NSA, OLI, and AWR).

    https://www.arm.gov/publications/tech_reports/handbooks/irt_handbook.pdf

    Author - Ken Kehoe

    Parameters
    ----------
    aeri_ds : Xarray Dataset Object
        The Dataset object containing AERI data.
    wnum_name : str
        The variable name for coordinate dimention of wave number Xarray Dataset.
    hatch_name : str or None
        The variable name for hatch status. If set to None will not try to set
        when hatch is not opent to NaN.
    rad_name : str
        The variable name for mean radiance in Xarray Dataset.
    tolerance : float
        The tolerance value to try and match for returned temperature.
    temp_low : float
        The initial low value to use in zbren function to invert radiances.
    temp_high : float
        The initial low value to use in zbren function to invert radiances.
    maxiter : int
        The maximum number if iterations to use with invertion process.
        Prevents runaway processes.

    Returns
    -------
    obj : Xarray Dataset Object or None
        The aeri_ds Dataset with new DataArray of temperatures added under
        variable name 'aeri_irt_equiv_temperature'.

    """
    # Get data values
    rf_wnum, rf = irt_response_function()
    wnum = aeri_ds[wnum_name].values
    mean_rad = aeri_ds[rad_name].values

    # Pull out AERI data for correct wavenumbers and apply response function --;
    index = np.where((wnum >= (rf_wnum[0] - 0.001)) & (wnum <= (rf_wnum[-1] + 0.001)))[0]
    if index.size == 0:
        raise ValueError('No wavenumbers match for aeri2irt')

    wnum = wnum[index]
    mean_rad = mean_rad[:, index]
    # If the wavenumbers in AERI data are not close enough to response function
    # match the wavenumbers and adjust.
    atol = 0.001
    if not np.all(np.isclose(wnum, rf_wnum, atol=atol)):
        index_wnum = []
        index_rf = []
        for ii in range(wnum.size):
            idx = (np.abs(wnum[ii] - rf_wnum)).argmin()
            if np.isclose(wnum[ii], rf_wnum[idx], atol=atol):
                index_wnum.append(ii)
                index_rf.append(idx)
        mean_rad = mean_rad[:, index_wnum]
        rf = rf[index_rf]

    # Apply response function to AERI radiance
    mean_rad = mean_rad * rf

    # Sum along wavenumber dimention to get a single value for each time step
    mean_rad = np.nansum(mean_rad, axis=1)

    # Loop over each time step of the AERI data and through the use of
    # solving for zero determine the AERI equivlante IRT sky temperature.
    aeri_irt_vals = np.full(mean_rad.size, np.nan, dtype=mean_rad.dtype)

    # Look for when the hatch is not in Open position and set values to NaN.
    if hatch_name is not None:
        flag_values = aeri_ds[hatch_name].attrs['flag_values']
        flag_meanings = aeri_ds[hatch_name].attrs['flag_meanings']
        if not isinstance(flag_meanings, list):
            flag_meanings = flag_meanings.split()
        flag_meanings = [att.lower() for att in flag_meanings]
        if not isinstance(flag_values, list):
            flag_values = flag_values.split()
            flag_values = [int(att) for att in flag_values]
        value = flag_values[flag_meanings.index('open')]
        mean_rad[aeri_ds[hatch_name].values != value] = np.nan

    for ii in range(mean_rad.size):
        if np.isnan(mean_rad[ii]):
            continue
        else:
            try:
                aeri_irt_vals[ii] = brentq(sum_function_irt, temp_low, temp_high,
                                           args=(mean_rad[ii], ), xtol=tolerance, maxiter=maxiter)
            except ValueError:
                pass

    # Add new values to Xarray Dataset
    aeri_ds['aeri_irt_equiv_temperature'] = (
        'time', aeri_irt_vals,
        {'long_name': 'Derived IRT equivalent temperatrues from AERI',
         'units': 'K'})

    return aeri_ds
