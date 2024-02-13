"""
Functions for radiosonde related calculations.

"""

import numpy as np
import xarray as xr
from operator import itemgetter
from itertools import groupby
import metpy.calc as mpcalc
from metpy.units import units

from act.utils.data_utils import convert_to_potential_temp


def calculate_precipitable_water(ds, temp_name='tdry', rh_name='rh', pres_name='pres'):
    """

    Function to calculate precipitable water vapor from ARM sondewnpn b1 data.
    Will first calculate saturation vapor pressure of all data using Arden-Buck
    equations, then calculate specific humidity and integrate over all pressure
    levels to give us a precipitable water value in centimeters.

    ds : xarray.Dataset
        Xarray dataset as read in by the ACT netCDF reader.
    temp_name : str
        Name of temperature field to use. Defaults to 'tdry' for sondewnpn b1
        level data.
    rh_name : str
        Name of relative humidity field to use. Defaults to 'rh' for sondewnpn
        b1 level data.
    pres_name : str
        Name of atmospheric pressure field to use. Defaults to 'pres' for
        sondewnpn b1 level data.

    """
    temp = ds[temp_name].values
    rh = ds[rh_name].values
    pres = ds[pres_name].values

    # Get list of temperature values for saturation vapor pressure calc
    temperature = []
    for t in np.nditer(temp):
        temperature.append(t)

    # Apply Arden-Buck equation to get saturation vapor pressure
    sat_vap_pres = []
    for t in temperature:
        # Over liquid water, above freezing
        if t >= 0:
            sat_vap_pres.append(0.61121 * np.exp((18.678 - (t / 234.5)) * (t / (257.14 + t))))
        # Over ice, below freezing
        else:
            sat_vap_pres.append(0.61115 * np.exp((23.036 - (t / 333.7)) * (t / (279.82 + t))))

    # convert rh from % to decimal
    rel_hum = []
    for r in np.nditer(rh):
        rel_hum.append(r / 100.0)

    # get vapor pressure from rh and saturation vapor pressure
    vap_pres = []
    for i in range(0, len(sat_vap_pres)):
        es = rel_hum[i] * sat_vap_pres[i]
        vap_pres.append(es)

    # Get list of pressure values for mixing ratio calc
    pressure = []
    for p in np.nditer(pres):
        pressure.append(p)

    # Mixing ratio calc

    mix_rat = []
    for i in range(0, len(vap_pres)):
        mix_rat.append(0.622 * vap_pres[i] / (pressure[i] - vap_pres[i]))

    # Specific humidity

    spec_hum = []
    for rat in mix_rat:
        spec_hum.append(rat / (1 + rat))

    # Integrate specific humidity

    pwv = 0.0
    for i in range(1, len(pressure) - 1):
        pwv = pwv + 0.5 * (spec_hum[i] + spec_hum[i - 1]) * (pressure[i - 1] - pressure[i])

    pwv = pwv / 0.098
    return pwv


def calculate_stability_indicies(
    ds,
    temp_name='temperature',
    td_name='dewpoint_temperature',
    p_name='pressure',
    moving_ave_window=0,
):
    """
    Function for calculating stability indices from sounding data.

    Parameters
    ----------
    ds : ACT dataset
        The dataset to compute the stability indicies of. Must have
        temperature, dewpoint, and pressure in vertical coordinates.
    temp_name : str
        The name of the temperature field.
    td_name : str
        The name of the dewpoint field.
    p_name : str
        The name of the pressure field.
    moving_ave_window : int
        Number of points to do a moving average on sounding data to reduce
        noise. This is useful if noise in the sounding is preventing parcel
        ascent.

    Returns
    -------
    ds : ACT dataset
        An ACT dataset with additional stability indicies added.

    """
    t = ds[temp_name]
    td = ds[td_name]
    p = ds[p_name]

    if not hasattr(t, 'units'):
        raise AttributeError('Temperature field must have units' + ' for ACT to discern!')

    if not hasattr(td, 'units'):
        raise AttributeError('Dewpoint field must have units' + ' for ACT to discern!')

    if not hasattr(p, 'units'):
        raise AttributeError('Pressure field must have units' + ' for ACT to discern!')
    if t.units == 'C':
        t_units = units.degC
    else:
        t_units = getattr(units, t.units)

    if td.units == 'C':
        td_units = units.degC
    else:
        td_units = getattr(units, td.units)

    p_units = getattr(units, p.units)

    # Sort all values by decreasing pressure
    t_sorted = np.array(t.values)
    td_sorted = np.array(td.values)
    p_sorted = np.array(p.values)
    ind_sort = np.argsort(p_sorted)
    t_sorted = t_sorted[ind_sort[-1:0:-1]]
    td_sorted = td_sorted[ind_sort[-1:0:-1]]
    p_sorted = p_sorted[ind_sort[-1:0:-1]]

    if moving_ave_window > 0:
        t_sorted = np.convolve(t_sorted, np.ones((moving_ave_window,)) / moving_ave_window)
        td_sorted = np.convolve(td_sorted, np.ones((moving_ave_window,)) / moving_ave_window)
        p_sorted = np.convolve(p_sorted, np.ones((moving_ave_window,)) / moving_ave_window)

    t_sorted = t_sorted * t_units
    td_sorted = td_sorted * td_units
    p_sorted = p_sorted * p_units

    t_profile = mpcalc.parcel_profile(p_sorted, t_sorted[0], td_sorted[0])

    # Calculate parcel trajectory
    ds['parcel_temperature'] = t_profile.magnitude
    ds['parcel_temperature'].attrs['units'] = t_profile.units

    # Calculate CAPE, CIN, LCL
    sbcape, sbcin = mpcalc.surface_based_cape_cin(p_sorted, t_sorted, td_sorted)

    lcl = mpcalc.lcl(p_sorted[0], t_sorted[0], td_sorted[0])
    try:
        lfc = mpcalc.lfc(p_sorted[0], t_sorted[0], td_sorted[0])
    except IndexError:
        lfc = np.nan * p_sorted.units

    mucape, mucin = mpcalc.most_unstable_cape_cin(p_sorted, t_sorted, td_sorted)

    where_500 = np.argmin(np.abs(p_sorted - 500 * units.hPa))
    li = t_sorted[where_500] - t_profile[where_500]

    ds['surface_based_cape'] = sbcape.magnitude
    ds['surface_based_cape'].attrs['units'] = 'J/kg'
    ds['surface_based_cape'].attrs['long_name'] = 'Surface-based CAPE'
    ds['surface_based_cin'] = sbcin.magnitude
    ds['surface_based_cin'].attrs['units'] = 'J/kg'
    ds['surface_based_cin'].attrs['long_name'] = 'Surface-based CIN'
    ds['most_unstable_cape'] = mucape.magnitude
    ds['most_unstable_cape'].attrs['units'] = 'J/kg'
    ds['most_unstable_cape'].attrs['long_name'] = 'Most unstable CAPE'
    ds['most_unstable_cin'] = mucin.magnitude
    ds['most_unstable_cin'].attrs['units'] = 'J/kg'
    ds['most_unstable_cin'].attrs['long_name'] = 'Most unstable CIN'
    ds['lifted_index'] = li.magnitude
    ds['lifted_index'].attrs['units'] = t_profile.units
    ds['lifted_index'].attrs['long_name'] = 'Lifted index'
    ds['level_of_free_convection'] = lfc.magnitude
    ds['level_of_free_convection'].attrs['units'] = lfc.units
    ds['level_of_free_convection'].attrs['long_name'] = 'Level of free convection'
    ds['lifted_condensation_level_temperature'] = lcl[1].magnitude
    ds['lifted_condensation_level_temperature'].attrs['units'] = lcl[1].units
    ds['lifted_condensation_level_temperature'].attrs[
        'long_name'
    ] = 'Lifted condensation level temperature'
    ds['lifted_condensation_level_pressure'] = lcl[0].magnitude
    ds['lifted_condensation_level_pressure'].attrs['units'] = lcl[0].units
    ds['lifted_condensation_level_pressure'].attrs[
        'long_name'
    ] = 'Lifted condensation level pressure'
    return ds


def calculate_pbl_liu_liang(
    ds,
    temperature='tdry',
    pressure='pres',
    windspeed='wspd',
    height='alt',
    smooth_height=3,
    land_parameter=True,
    llj_max_alt=1500.0,
    llj_max_wspd=2.0,
):
    """
    Function for calculating the PBL height from a radiosonde profile
    using the Liu-Liang 2010 technique.  There are some slight descrepencies
    in the function from the ARM implementation 1.) it imposes a 1500m (keyword)
    height on the definition of the LLJ and 2.) the interpolation is slightly different
    using python functions

    Parameters
    ----------
    ds : xarray Dataset
        Dataset housing radiosonde profile for calculations
    temperature : str
        The name of the temperature field.
    pressure : str
        The name of the pressure field.
    windspeed : str
        The name of the  wind speed field.
    height : str
        The name of the height field
    smooth_height : int
        Number of points to do a moving average on sounding height data to reduce noise
    land_parameter : boolean
        Set to True if retrievals over land or false to retrievals over water
    llj_max_alt : float
        Maximum altitude the LLJ 2 m/s difference should be checked against
    llj_max_wspd : float
        Maximum wind speed threshold to use to define LLJ

    Returns
    -------
    ds : xarray Dataset
        xarray dataset with results stored in pblht_liu_liang variable

    References
    ----------
    Liu, Shuyan, and Xin-Zhong Liang. "Observed diurnal cycle climatology of planetary
        boundary layer height." Journal of Climate 23, no. 21 (2010): 5790-5809.

    Sivaraman, C., S. McFarlane, E. Chapman, M. Jensen, T. Toto, S. Liu, and M. Fischer.
        "Planetary boundary layer (PBL) height value added product (VAP): Radiosonde retrievals."
        Department of Energy Office of Science Atmospheric Radiation Measurement (ARM) Program
        (United States) (2013).

    """

    # Preprocess the sonde data to ensure the same methods across all retrievals
    ds2 = preprocess_sonde_data(
        ds,
        temperature=temperature,
        pressure=pressure,
        height=height,
        smooth_height=smooth_height,
        base=5.0,
    )

    pres = ds2[pressure].values
    wspd = ds2[windspeed].values
    alt = ds2[height].values

    theta = ds2['potential_temperature'].values

    # Calculate the lapse rate
    theta_gradient = np.diff(theta) / np.diff(alt)

    # Calculate AGL
    if np.isnan(alt[0]):
        idx = np.where(~np.isnan(alt))[0]
        agl = alt - alt[idx[0]]
    else:
        agl = alt - alt[0]

    theta_diff = theta[4] - theta[1]
    theta_gradient = np.diff(theta) / np.diff(alt / 1000.0)

    # Set up threshold values
    if land_parameter:
        stability_thresh = 1.0  # K
        inst_thresh = 0.5  # K
        overshoot_thresh = 4.0  # K/km
    else:
        stability_thresh = 0.2  # K
        inst_thresh = 0.1  # K
        overshoot_thresh = 0.5  # K/km

    # Check Regimes
    if theta_diff < 0 - stability_thresh:
        regime = 'CBL'
    if theta_diff > abs(stability_thresh):
        regime = 'SBL'
    if (0 - stability_thresh) <= theta_diff <= abs(stability_thresh):
        regime = 'NRL'

    # Calculate for CBL/NRL regimes
    pbl_stable = np.nan
    pbl_shear = np.nan

    if regime == 'CBL' or regime == 'NRL':
        # Calculate gradient from first level
        theta_gradient_0 = theta - theta[0]

        # Only process data above 150m ARM
        idx = np.where(agl > 150)[0][0]
        theta_gradient_0[0:idx] = np.nan

        # Scan upward to find lowest level that meets condition
        idx = np.where(theta_gradient_0 >= inst_thresh)[0]
        theta_gradient[0 : idx[0]] = np.nan

        # Scan upward from previous level to search for overlying inversion layer
        idx = np.where(theta_gradient >= overshoot_thresh)[0]
        pbl = alt[idx[0]]
    else:
        idx = np.array(
            [
                i
                for i, t in enumerate(theta_gradient[1:-1])
                if theta_gradient[i] < theta_gradient[i - 1]
                and theta_gradient[i] < theta_gradient[i + 1]
            ]
        )

        for i in idx:
            cond1 = (theta_gradient[i] - theta_gradient[i - 1]) < -40.0
            cond2 = (theta_gradient[i + 1] < overshoot_thresh) or (
                theta_gradient[i + 2] < overshoot_thresh
            )
            if cond1 or cond2:
                # This gets the ARM answer
                pbl_stable = (alt[i + 1] + alt[i]) / 2.0
                # pbl_stable = alt[i]
                break

        # Check for low-level jet
        # Find the height of the maximum windspeed and look up to find layer 2m/s lower
        # Stull 1988 indicates LLJ is defined as where there is a relative wind speed
        # maximum that is more than 2 m/s faster than the wind speeds above it within
        # the lowest 1500m of the atmosphere. Keywords to adjust are provided
        idh = np.where(alt <= llj_max_alt)[0]
        max_wspd_ind = [i for i, w in enumerate(wspd[:-1]) if wspd[i] > wspd[i + 1]][0]
        diff = wspd[max_wspd_ind] - wspd[max_wspd_ind : idh[-1]]
        idx = np.where(diff > llj_max_wspd)[0]
        if len(idx) > 0:
            wspd_to_surf = np.diff(np.flip(wspd[0:max_wspd_ind]))
            wspd_monotonic = np.all(wspd_to_surf <= 0.0)
            if wspd_monotonic:
                pbl_shear = alt[max_wspd_ind]

        if ~np.all(np.isnan([pbl_stable, pbl_shear])):
            pbl = np.nanmin([pbl_stable, pbl_shear])
        else:
            pbl = -9999.0

    atts = {'units': 'm', 'long_name': 'Planteary Boundary Layer Height Liu-Liang'}
    da = xr.DataArray(pbl, attrs=atts)
    ds['pblht_liu_liang'] = da

    atts = {
        'units': '',
        'long_name': 'Planteary Boundary Layer Regime Classification Liu-Liang',
    }
    da = xr.DataArray(regime, attrs=atts)
    ds['pblht_regime_liu_liang'] = da

    atts = {'units': 'mb', 'long_name': 'Gridded pressure'}
    da = xr.DataArray(pres, coords={'atm_pres_ss': pres}, dims=['atm_pres_ss'], attrs=atts)
    ds['atm_pres_ss'] = da

    atts = {'units': 'K', 'long_name': 'Gridded potential temperature'}
    da = xr.DataArray(theta, coords={'atm_pres_ss': pres}, dims=['atm_pres_ss'], attrs=atts)
    ds['potential_temperature_ss'] = da

    atts = {'units': 'm', 'long_name': 'Gridded altitude'}
    da = xr.DataArray(alt, coords={'atm_pres_ss': pres}, dims=['atm_pres_ss'], attrs=atts)
    ds['alt_ss'] = da

    atts = {'units': 'm', 'long_name': 'PBL Stable Condition 1'}
    da = xr.DataArray(pbl_stable, attrs=atts)
    ds['pblht_liu_liang_stable_cond'] = da

    atts = {'units': 'm', 'long_name': 'PBL Shear Condition 2'}
    da = xr.DataArray(pbl_shear, attrs=atts)
    ds['pblht_liu_liang_shear_cond'] = da

    return ds


def calculate_pbl_heffter(
    ds,
    temperature='tdry',
    pressure='pres',
    height='alt',
    smooth_height=3,
    base=5.0,
):
    """
    Function for calculating the PBL height from a radiosonde profile
    using the Heffter technique. There are differences from the ARM
    VAP at times due to different averaging schemes.  Larger differences
    do occur at times and are unknown as to the cause but it is being
    investigated and is potential a code issue with the VAP.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset housing radiosonde profile for calculations
    temperature : str
        The name of the temperature field.
    pressure : str
        The name of the pressure field.
    height : str
        The name of the height field
    smooth_height : int
        Number of points to do a moving average on sounding height data to reduce noise
    base : int
        Interval for pressure gridding.  In testing, 5 mb was found to produce results with
        the lowest RMS

    Returns
    -------
    ds : xarray Dataset
        xarray dataset with results stored in pblht_liu_liang variable

    References
    ----------
    Heffter JL. 1980. “Transport Layer Depth Calculations.” Second Joint Conference on
        Applications of Air Pollution Meteorology, New Orleans, Louisiana.

    Sivaraman, C., S. McFarlane, E. Chapman, M. Jensen, T. Toto, S. Liu, and M. Fischer.
        "Planetary boundary layer (PBL) height value added product (VAP): Radiosonde retrievals."
        Department of Energy Office of Science Atmospheric Radiation Measurement (ARM) Program
        (United States) (2013).

    """

    # Preprocess the sonde data to ensure the same methods across all retrievals
    ds2 = preprocess_sonde_data(
        ds,
        temperature=temperature,
        pressure=pressure,
        height=height,
        smooth_height=smooth_height,
        base=base,
    )

    # Get data
    pres = ds2[pressure].values
    alt = ds2[height].values
    theta = ds2['potential_temperature'].values

    # Calculate the lapse rate
    theta_gradient = np.diff(theta) / np.diff(alt)

    # Calculate AGL
    if np.isnan(alt[0]):
        idx = np.where(~np.isnan(alt))[0]
        agl = alt - alt[idx[0]]
    else:
        agl = alt - alt[0]

    # Find where the lapse rate is greater than 0.005 K/m
    idx = np.where(theta_gradient >= 0.005)[0]

    # Find the consistent layers by grouping the indices together
    # Does not include a single height as a layer
    ranges = []
    for key, group in groupby(enumerate(idx), lambda i: i[0] - i[1]):
        group = list(map(itemgetter(1), group))
        if group[-1] - group[0] > 0:
            ranges.append((group[0], group[-1]))

    # Subset ranges to lowest 5
    if len(ranges) > 5:
        ranges = ranges[0:5]

    # For each layer, calculate the difference in theta from
    # top and bottom of the layer.  The lowest layer where the
    # difference is > 2 K is set as the PBL.
    pbl = 0.0
    theta_diff_layer = []
    bottom_inversion = []
    top_inversion = []
    for r in ranges:
        if agl[r[1]] > 4000.0:
            continue
        theta_diff = theta[r[1]] - theta[r[0]]
        theta_diff_layer.append(theta_diff)
        bottom_inversion.append(alt[r[0]])
        top_inversion.append(alt[r[1]])
        if pbl == 0.0 and theta_diff > 2.0:
            pbl = alt[r[0]]

    if len(theta_diff_layer) == 0:
        pbl = -9999.0

    # If PBL is not set, set it to the layer with the max theta diff
    if pbl == 0.0:
        idx = np.argmax(theta_diff_layer)
        pbl = bottom_inversion[idx]

    # Add variables to the dataset
    atts = {'units': 'm', 'long_name': 'Planteary Boundary Layer Height Heffter'}
    da = xr.DataArray(pbl, attrs=atts)
    ds['pblht_heffter'] = da

    atts = {'units': 'mb', 'long_name': 'Gridded pressure'}
    da = xr.DataArray(pres, coords={'atm_pres_ss': pres}, dims=['atm_pres_ss'], attrs=atts)
    ds['atm_pres_ss'] = da

    atts = {'units': 'K', 'long_name': 'Gridded potential temperature'}
    da = xr.DataArray(theta, coords={'atm_pres_ss': pres}, dims=['atm_pres_ss'], attrs=atts)
    ds['potential_temperature_ss'] = da

    atts = {'units': 'm', 'long_name': 'Gridded altitude'}
    da = xr.DataArray(alt, coords={'atm_pres_ss': pres}, dims=['atm_pres_ss'], attrs=atts)
    ds['alt_ss'] = da

    atts = {'units': 'm', 'long_name': 'Bottom height of inversion layers'}
    da = xr.DataArray(
        bottom_inversion,
        coords={'layers': list(range(len(bottom_inversion)))},
        dims=['layers'],
        attrs=atts,
    )
    ds['bottom_inversion'] = da

    atts = {'units': 'm', 'long_name': 'Top height of inversion layers'}
    da = xr.DataArray(
        top_inversion,
        coords={'layers': list(range(len(top_inversion)))},
        dims=['layers'],
        attrs=atts,
    )
    ds['top_inversion'] = da

    return ds


def preprocess_sonde_data(
    ds,
    temperature='tdry',
    pressure='pres',
    height='alt',
    smooth_height=3,
    base=5.0,
):
    """
    Function for processing the SONDE data for the PBL calculations.
    This is to ensure consistency and also applies some QC to the
    processing.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset housing radiosonde profile for calculations
    temperature : str
        The name of the temperature field.
    pressure : str
        The name of the pressure field.
    height : str
        The name of the height field
    smooth_height : int
        Number of points to do a moving average on sounding height data to reduce noise
    base : int
        Interval for pressure gridding.  In testing, 5 mb was found to produce results with
        the lowest RMS

    Returns
    -------
    ds : xarray.Dataset
        Xarray dataset containing processed sonde data.

    References
    ----------
    Sivaraman, C., S. McFarlane, E. Chapman, M. Jensen, T. Toto, S. Liu, and M. Fischer.
        "Planetary boundary layer (PBL) height value added product (VAP): Radiosonde retrievals."
        Department of Energy Office of Science Atmospheric Radiation Measurement (ARM) Program
        (United States) (2013).

    """

    # Get the initial time and temp values
    time_0 = ds['time'].values
    temp_0 = ds[temperature].values

    # Apply a rolling average to smooth the pressure out
    ds[pressure] = ds[pressure].rolling(time=smooth_height, min_periods=1, center=True).mean()

    # Swap time and pressure for doing the appropriate gridding
    ds2 = ds.swap_dims(dims_dict={'time': pressure})
    for var in ds2:
        ds2[var].attrs = ds2[var].attrs

    # Set up the pressure grid
    starting_pres = base * np.ceil(float(ds2[pressure].values[2]) / base)
    p_grid = np.flip(np.arange(100.0, starting_pres + base, base))

    # Pull out the data that's nearest to the pressure grid.  If it errors
    # it will smooth the data more.  This tends to happen if there are multiple
    # values of pressure that are the same
    try:
        ds2 = ds2.sel({pressure: p_grid}, method='nearest')
    except Exception:
        ds[pressure] = (
            ds[pressure].rolling(time=smooth_height + 4, min_periods=2, center=True).mean()
        )
        ds2 = ds.swap_dims(dims_dict={'time': pressure})
        for var in ds2:
            ds2[var].attrs = ds[var].attrs
        try:
            ds2 = ds2.sel({pressure: p_grid}, method='nearest')
        except Exception:
            raise ValueError('Sonde profile does not have unique pressures after smoothing')

    # Get data
    alt = ds2[height].values
    pres = ds2[pressure].values
    temp = ds2[temperature].values

    # Perform Pre-processing checks
    if len(temp) == 0.0:
        raise ValueError('No data in profile')

    if np.nanmax(alt) < 1000.0:
        raise ValueError('Max altitude < 1000m')

    if np.nanmax(pres) <= 200.0:
        raise ValueError('Max pressure <= 200 hPa')

    # Check temperature delta
    t1 = time_0[0]
    t2 = t1 + np.timedelta64(10, 's')
    idx = np.where((time_0 >= t1) & (time_0 <= t2))[0]
    t_delta = abs(temp_0[idx[-1]] - temp_0[idx[0]])
    if t_delta > 30.0:
        raise ValueError('Temperature changes by >30º in first 10 seconds')

    # Check min/max
    if np.nanmax(temp) > 50.0 or np.nanmin(temp) < -90:
        raise ValueError('Temperature outside acceptable range (-90, 50)')

    if np.isnan(pres[0]) or np.isnan(pres[1]):
        raise ValueError('First two pressure values bad')

    # Calculate potential temperature and subsequent gradients
    theta = (
        convert_to_potential_temp(ds=ds2, temp_var_name=temperature, press_var_name=pressure)
        + 273.15
    )

    # Set variables to return
    atts = {'units': 'K', 'long_name': 'Potential temperature'}
    da = xr.DataArray(theta, coords=ds2['tdry'].coords, dims=ds2[temperature].dims, attrs=atts)
    ds2['potential_temperature'] = da

    return ds2
