"""
Functions for solar radiation related calculations and retrievals.

"""

import numpy as np
import xarray as xr
from scipy.constants import Stefan_Boltzmann
from act.utils.datetime_utils import datetime64_to_datetime
from act.utils.geo_utils import get_solar_azimuth_elevation


def calculate_dsh_from_dsdh_sdn(obj, dsdh='down_short_diffuse_hemisp',
                                sdn='short_direct_normal', lat='lat',
                                lon='lon'):
    """

    Function to derive the downwelling shortwave hemispheric irradiance from the
    downwelling shortwave diffuse hemispheric irradiance (dsdh) and the shortwave
    direct normal irradiance (sdn) at a given location (lat,lon)

    Parameters
    ----------
    obj : Xarray dataset
        Object where variables for these calculations are stored
    dsdh : str
        Name of the downwelling shortwave diffuse hemispheric irradiance field to use.
        Defaults to downwelling_sw_diffuse_hemisp_irradiance.
    sdn : str
        Name of shortwave direct normal irradiance field to use.
        Defaults to shortwave_direct_normal_irradiance.
    lat : str
        Name of latitude field in dataset to use. Defaults to 'lat'.
    lon : str
        Name of longitued field in dataset to use. Defaults to 'lon'.

    Returns
    -------

    obj: Xarray dataset
        ACT Xarray dataset oject with calculations included as new variables.

    """

    # Calculating Derived Down Short Hemisp
    tt = datetime64_to_datetime(obj['time'].values)
    elevation, _, _ = get_solar_azimuth_elevation(obj[lat].values, obj[lon].values, tt)
    solar_zenith = np.cos(np.radians(90. - elevation))
    dsh = (obj[dsdh].values + (solar_zenith * obj[sdn].values))

    # Add data back to object
    atts = {'long_name': 'Derived Downwelling Shortwave Hemispheric Irradiance', 'units': 'W/m^2'}
    da = xr.DataArray(dsh, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
    obj['derived_down_short_hemisp'] = da

    return obj


def calculate_irradiance_stats(obj, variable=None, variable2=None, diff_output_variable=None,
                               ratio_output_variable=None, threshold=None):
    """

    Function to calculate the difference and ratio between two irradiance.

    Parameters
    ----------
    obj : ACT object
        Object where variables for these calculations are stored
    variable : str
        Name of the first irradiance variable
    variable2 : str
        Name of the second irradiance variable
    diff_output_variable : str
        Variable name to store the difference results
        Defaults to 'diff[underscore]'+variable
    ratio_output_variable : str
        Variable name to store the ratio results
        Defaults to 'ratio[underscore]'+variable

    Returns
    -------

    obj: ACT Object
        Object with calculations included as new variables.

    """

    if variable is None or variable2 is None:
        return obj
    if diff_output_variable is None:
        diff_output_variable = 'diff_' + variable
    if ratio_output_variable is None:
        ratio_output_variable = 'ratio_' + variable

    # ---------------------------------
    # Calculating Difference
    # ---------------------------------
    diff = obj[variable] - obj[variable2]
    atts = {'long_name': ' '.join(['Difference between', variable, 'and', variable2]), 'units': 'W/m^2'}
    da = xr.DataArray(diff, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
    obj[diff_output_variable] = da

    # ---------------------------------
    # Calculating Irradiance Ratio
    # ---------------------------------
    ratio = obj[variable].values / obj[variable2].values
    if threshold is not None:
        index = np.where((obj[variable].values < threshold) & (obj[variable2].values < threshold))
        ratio[index] = np.nan

    atts = {'long_name': ' '.join(['Ratio between', variable, 'and', variable2]), 'units': ''}
    da = xr.DataArray(ratio, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
    obj[ratio_output_variable] = da

    return obj


def calculate_net_radiation(obj, ush='up_short_hemisp', ulh='up_long_hemisp', dsh='down_short_hemisp',
                            dlhs='down_long_hemisp_shaded', smooth=None):

    """

    Function to calculate the net  radiation from upwelling short and long-wave irradiance and
    downwelling short and long-wave hemisperic irradiances

    Parameters
    ----------
    obj : ACT object
        Object where variables for these calculations are stored
    ush : str
        Name of the upwelling shortwave hemispheric variable
    ulh : str
        Name of the upwelling longwave hemispheric variable
    dsh : str
        Name of the downwelling shortwave hemispheric variable
    dlhs : str
        Name of the downwelling longwave hemispheric variable
    smooth : int
        Smoothing to apply to the net radiation.  This will create an additional variable

    Returns
    -------

    obj: ACT Object
        Object with calculations included as new variables.

    """

    # Calculate Net Radiation
    ush_da = obj[ush]
    ulh_da = obj[ulh]
    dsh_da = obj[dsh]
    dlhs_da = obj[dlhs]

    net = -ush_da + dsh_da - ulh_da + dlhs_da

    atts = {'long_name': 'Calculated Net Radiation', 'units': 'W/m^2'}
    da = xr.DataArray(net, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
    obj['net_radiation'] = da

    if smooth is not None:
        net_smoothed = net.rolling(time=smooth).mean()
        atts = {'long_name': 'Net Radiation Smoothed by ' + str(smooth), 'units': 'W/m^2'}
        da = xr.DataArray(net_smoothed, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
        obj['net_radiation_smoothed'] = da

    return obj


def calculate_longwave_radiation(obj, temperature_var=None, vapor_pressure_var=None, met_obj=None,
                                 emiss_a=0.61, emiss_b=0.06):

    """

    Function to calculate longwave radiation during clear and cloudy sky conditions
    using equations from Monteith and Unsworth 2013, Prata 1996, as reported in
    Splitt and Bahrmann 1999.

    Parameters
    ----------
    obj : ACT object
        Object where variables for these calculations are stored
    temperature_var : str
        Name of the temperature variable to use
    vapor_pressure_var : str
        Name of the vapor pressure variable to use
    met_obj : ACT object
        Object where surface meteorological variables for these calculations are stored
        if not given, will assume they are in the main object passed in
    emiss_a : float
        a coefficient for the emissivity calculation of e = a + bT
    emiss_b : float
        a coefficient for the emissivity calculation of e = a + bT

    Returns
    -------
    obj : ACT object
        ACT object with 3 new variables; monteith_clear, monteith_cloudy, prata_clear

    References
    ---------
    Monteith, John L., and Mike H. Unsworth. 2013. Principles of Environmental Physics.
        Edited by John L. Monteith and Mike H. Unsworth. Boston: Academic Press.

    Prata, A. J. 1996. “A New Long-Wave Formula for Estimating Downward Clear-Sky Radiation at
        the Surface.” Quarterly Journal of the Royal Meteorological Society 122 (533): 1127–51.

    Splitt, M. E., and C. P. Bahrmann. 1999. Improvement in the Assessment of SIRS Broadband
        Longwave Radiation Data Quality. Ninth ARM Science Team Meeting Proceedings,
        San Antonio, Texas, March 22-26

    """
    if met_obj is not None:

        T = met_obj[temperature_var] + 273.15  # C to K
        e = met_obj[vapor_pressure_var] * 10.  # kpa to hpa
    else:
        T = obj[temperature_var] + 273.15  # C to K
        e = obj[vapor_pressure_var] * 10.  # kpa to hpa

    if len(T) == 0 or len(e) == 0:
        raise ValueError('Temperature and Vapor Pressure are Needed')

    # Get Stefan Boltzmann Constant
    stefan = Stefan_Boltzmann

    # Calculate sky emissivity from Splitt and Bahrmann 1999
    esky = emiss_a + emiss_b * np.sqrt(e)

    # Base clear sky longwave calculation from Monteith 2013
    lw_calc_clear = esky * stefan * T**4

    # Prata 1996 Calculation
    xi = 46.5 * (e / T)
    lw_calc_clear_prata = (1.0 - (1.0 + xi) * np.exp(-(1.2 + 3.0 * xi)**.5)) * stefan * T**4

    # Monteith Cloudy Calcuation as indicated by Splitt and Bahrmann 1999
    lw_calc_cldy = esky * (1.0 + (0.178 - 0.00957 * (T - 290.))) * stefan * T**4

    atts = {'long_name': 'Clear Sky Estimate-(Monteith, 1973)', 'units': 'W/m^2'}
    da = xr.DataArray(lw_calc_clear, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
    obj['monteith_clear'] = da

    atts = {'long_name': 'Overcast Sky Estimate-(Monteith, 1973)', 'units': 'W/m^2'}
    da = xr.DataArray(lw_calc_cldy, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
    obj['monteith_cloudy'] = da

    atts = {'long_name': 'Clear Sky Estimate-(Prata, 1996)', 'units': 'W/m^2'}
    da = xr.DataArray(lw_calc_clear_prata, coords={'time': obj['time'].values}, dims=['time'], attrs=atts)
    obj['prata_clear'] = da

    return obj
