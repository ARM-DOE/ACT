"""
Functions for solar radiation related calculations and retrievals.

"""

import numpy as np
import xarray as xr
from scipy.constants import Stefan_Boltzmann

from act.utils.geo_utils import get_solar_azimuth_elevation
from act.utils.data_utils import convert_units


def calculate_dsh_from_dsdh_sdn(
    ds,
    dsdh='down_short_diffuse_hemisp',
    sdn='short_direct_normal',
    lat='lat',
    lon='lon',
):
    """

    Function to derive the downwelling shortwave hemispheric irradiance (dsh) from the
    downwelling shortwave diffuse hemispheric irradiance (dsdh) and the shortwave
    direct normal irradiance (sdn). The derived values are added the returned
    Datasets as a new varible.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset where variables for these calculations are stored.
    dsdh : str
        Name of the downwelling shortwave diffuse hemispheric irradiance field to use.
    sdn : str
        Name of shortwave direct normal irradiance field to use.
    lat : str
        Name of latitude variable in dataset to use for deriving solar zenight angle.
    lon : str
        Name of longitude variable in dataset to use for deriving solar zenight angle.

    Returns
    -------

    ds: xarray.Dataset
        ACT Xarray Dataset with calculations included as new variable.

    """

    ds = calculate_ghi_from_dni_dhi(ds, dni=sdn, dhi=dsdh, lat=lat, lon=lon)

    return ds


def calculate_ghi_from_dni_dhi(
    ds,
    dni='short_direct_normal',
    dhi='down_short_diffuse_hemisp',
    derived='derived_down_short_hemisp',
    long_name='Derived global horizontal irradiance',
    solar_zenith=None,
    lat='lat',
    lon='lon',
):
    """

    Function to derive the Global Horizontal Irradiance (GHI) from
    Direct Normal Irradiance (DNI) and Diffuse Horizontal
    Irradiance (DHI). The derived values are added the returned Datasets as a new DataArray.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset where variables used for these calculations are stored
    dni : str
        Name of the direct normal irradiance DataArray to use.
    dhi : str
        Name of diffuse hemispheric irradiance DataArray to use.
    derived : str
        Name of new diffuse horizontal irradiance DataArray.
    long_name : str
        Long name used in new DataArray.
    solar_zenith : str or None
        Name of solar zenith DataArray in Dataset. If set to None will calculate using
        location and time variables from Dataset.
    lat : str
        Name of latitude DataArray in dataset to use for deriving solar zenight angle.
        Ignored if solar_zenith provided.
    lon : str
        Name of longitued DataArray in dataset to use for deriving solar zenight angle.
        Ignored if solar_zenith provided.

    Returns
    -------

    ds: xarray.Dataset
        ACT Xarray Dataset with global horizontal irradiance included as new DataArray.

    """

    # Get solar zenith angle
    if solar_zenith is not None:
        sz = ds[solar_zenith].values
        sz = convert_units(sz, ds[solar_zenith].attrs['units'], 'radians')
        cos_sz = np.cos(sz)
    else:
        elevation, _, _ = get_solar_azimuth_elevation(
            ds[lat].values, ds[lon].values, ds['time'].values
        )
        cos_sz = np.cos(np.radians(90.0 - elevation))

    ghi = ds[dhi].values + (cos_sz * ds[dni].values)

    # Add data into Dataset
    ds[derived] = xr.DataArray(
        ghi,
        dims=['time'],
        attrs={
            'long_name': long_name,
            'units': ds[dhi].attrs['units'],
        },
    )

    return ds


def calculate_dni_from_dhi_ghi(
    ds,
    dhi='down_short_diffuse_hemisp',
    ghi='down_short_hemisp',
    derived='derived_short_direct_normal',
    long_name='Derived direct normal irradiance',
    solar_zenith=None,
    lat='lat',
    lon='lon',
):
    """

    Function to derive the Direct Normal Irradiance (DNI) from
    Diffuse Horizontal Irradiance (DHI) and Global Horizontal
    Irradiance (GHI). The derived values are added the returned Datasets as a new DataArray.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset where variables used for these calculations are stored
    dhi : str
        Name of the diffuse horizontal irradiance DataArray to use.
    ghi : str
        Name of global hemispheric irradiance DataArray to use.
    derived : str
        Name of new direct normal irradiance DataArray.
    long_name : str
        Long name used in new DataArray.
    solar_zenith : str or None
        Name of solar zenith DataArray in Dataset. If set to None will calculate using
        location and time variables from Dataset.
    lat : str
        Name of latitude DataArray in dataset to use for deriving solar zenight angle.
        Ignored if solar_zenith provided.
    lon : str
        Name of longitued DataArray in dataset to use for deriving solar zenight angle.
        Ignored if solar_zenith provided.

    Returns
    -------

    ds: xarray.Dataset
        ACT Xarray Dataset with direct normal irradiance included as new DataArray.

    """

    # Get solar zenith angle
    if solar_zenith is not None:
        sz = ds[solar_zenith].values
        sz = convert_units(sz, ds[solar_zenith].attrs['units'], 'radians')
        cos_sz = np.cos(sz)
    else:
        elevation, _, _ = get_solar_azimuth_elevation(
            ds[lat].values, ds[lon].values, ds['time'].values
        )
        cos_sz = np.cos(np.radians(90.0 - elevation))

    dni = (ds[ghi].values - ds[dhi].values) / cos_sz

    # Add data into Dataset
    ds[derived] = xr.DataArray(
        dni,
        dims=['time'],
        attrs={
            'long_name': long_name,
            'units': ds[dhi].attrs['units'],
        },
    )

    return ds


def calculate_dhi_from_dni_ghi(
    ds,
    dni='short_direct_normal',
    ghi='down_short_hemisp',
    derived='derived_down_short_diffuse_hemisp',
    long_name='Derived diffuse horizontal irradiance',
    solar_zenith=None,
    lat='lat',
    lon='lon',
):
    """

    Function to derive the Diffuse Horizontal Irradiance (DHI) from
    Direct Normal Irradiance (DNI) and Global Horizontal
    Irradiance (GHI). The derived values are added the returned Datasets as a new DataArray.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset where variables used for these calculations are stored
    dni : str
        Name of the dirret normal irradiance DataArray to use.
    ghi : str
        Name of global hemispheric irradiance DataArray to use.
    derived : str
        Name of new diffuse horizontal irradiance DataArray.
    long_name : str
        Long name used in new DataArray.
    solar_zenith : str or None
        Name of solar zenith DataArray in Dataset. If set to None will calculate using
        location and time variables from Dataset.
    lat : str
        Name of latitude DataArray in dataset to use for deriving solar zenight angle.
        Ignored if solar_zenith provided.
    lon : str
        Name of longitued DataArray in dataset to use for deriving solar zenight angle.
        Ignored if solar_zenith provided.

    Returns
    -------

    ds: xarray.Dataset
        ACT Xarray Dataset with diffuse horizontal irradiance included as new DataArray.

    """

    # Get solar zenith angle
    if solar_zenith is not None:
        sz = ds[solar_zenith].values
        sz = convert_units(sz, ds[solar_zenith].attrs['units'], 'radians')
        cos_sz = np.cos(sz)
    else:
        elevation, _, _ = get_solar_azimuth_elevation(
            ds[lat].values, ds[lon].values, ds['time'].values
        )
        cos_sz = np.cos(np.radians(90.0 - elevation))

    dhi = ds[ghi].values - (ds[dni].values * cos_sz)

    # Add data into Dataset
    ds[derived] = xr.DataArray(
        dhi,
        dims=['time'],
        attrs={
            'long_name': long_name,
            'units': ds[ghi].attrs['units'],
        },
    )

    return ds


def calculate_irradiance_stats(
    ds,
    variable=None,
    variable2=None,
    diff_output_variable=None,
    ratio_output_variable=None,
    threshold=None,
):
    """

    Function to calculate the difference and ratio between two irradiance.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset where variables for these calculations are stored
    variable : str
        Name of the first irradiance variable
    variable2 : str
        Name of the second irradiance variable
    diff_output_variable : str
        Variable name to store the difference results
        Defaults to 'diff_' + variable
    ratio_output_variable : str
        Variable name to store the ratio results
        Defaults to 'ratio_' + variable

    Returns
    -------

    ds : xarray.Dataset
        Xarray dataset with calculations included as new variables.

    """

    if variable is None or variable2 is None:
        return ds

    if diff_output_variable is None:
        diff_output_variable = 'diff_' + variable

    if ratio_output_variable is None:
        ratio_output_variable = 'ratio_' + variable

    # ---------------------------------
    # Calculating Difference
    # ---------------------------------
    diff = ds[variable] - ds[variable2]
    atts = {
        'long_name': ' '.join(['Difference between', variable, 'and', variable2]),
        'units': 'W/m^2',
    }
    da = xr.DataArray(diff, coords={'time': ds['time'].values}, dims=['time'], attrs=atts)
    ds[diff_output_variable] = da

    # ---------------------------------
    # Calculating Irradiance Ratio
    # ---------------------------------
    ratio = ds[variable].values / ds[variable2].values
    if threshold is not None:
        index = np.where((ds[variable].values < threshold) & (ds[variable2].values < threshold))
        ratio[index] = np.nan

    atts = {
        'long_name': ' '.join(['Ratio between', variable, 'and', variable2]),
        'units': '',
    }
    da = xr.DataArray(ratio, coords={'time': ds['time'].values}, dims=['time'], attrs=atts)
    ds[ratio_output_variable] = da

    return ds


def calculate_net_radiation(
    ds,
    ush='up_short_hemisp',
    ulh='up_long_hemisp',
    dsh='down_short_hemisp',
    dlhs='down_long_hemisp_shaded',
    smooth=None,
):
    """

    Function to calculate the net radiation from upwelling short and long-wave irradiance and
    downwelling short and long-wave hemisperic irradiances

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset where variables for these calculations are stored
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

    ds : xarray.Dataset
        Xarray dataset with calculations included as new variables.

    """

    # Calculate Net Radiation
    ush_da = ds[ush]
    ulh_da = ds[ulh]
    dsh_da = ds[dsh]
    dlhs_da = ds[dlhs]

    net = -ush_da + dsh_da - ulh_da + dlhs_da

    atts = {'long_name': 'Calculated Net Radiation', 'units': 'W/m^2'}
    da = xr.DataArray(net, coords={'time': ds['time'].values}, dims=['time'], attrs=atts)
    ds['net_radiation'] = da

    if smooth is not None:
        net_smoothed = net.rolling(time=smooth).mean()
        atts = {
            'long_name': 'Net Radiation Smoothed by ' + str(smooth),
            'units': 'W/m^2',
        }
        da = xr.DataArray(
            net_smoothed, coords={'time': ds['time'].values}, dims=['time'], attrs=atts
        )
        ds['net_radiation_smoothed'] = da

    return ds


def calculate_longwave_radiation(
    ds,
    temperature_var=None,
    vapor_pressure_var=None,
    met_ds=None,
    emiss_a=0.61,
    emiss_b=0.06,
):
    """

    Function to calculate longwave radiation during clear and cloudy sky conditions
    using equations from Monteith and Unsworth 2013, Prata 1996, as reported in
    Splitt and Bahrmann 1999.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset where variables for these calculations are stored
    temperature_var : str
        Name of the temperature variable to use
    vapor_pressure_var : str
        Name of the vapor pressure variable to use
    met_ds : xarray.Dataset
        Xarray dataset where surface meteorological variables for these calculations are
        stored if not given, will assume they are in the main dataset passed in
    emiss_a : float
        a coefficient for the emissivity calculation of e = a + bT
    emiss_b : float
        a coefficient for the emissivity calculation of e = a + bT

    Returns
    -------
    ds : xarray.Dataset
        Xarray dataset with 3 new variables; monteith_clear, monteith_cloudy, prata_clear

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
    if met_ds is not None:
        T = met_ds[temperature_var] + 273.15  # C to K
        e = met_ds[vapor_pressure_var] * 10.0  # kpa to hpa
    else:
        T = ds[temperature_var] + 273.15  # C to K
        e = ds[vapor_pressure_var] * 10.0  # kpa to hpa

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
    lw_calc_clear_prata = (1.0 - (1.0 + xi) * np.exp(-((1.2 + 3.0 * xi) ** 0.5))) * stefan * T**4

    # Monteith Cloudy Calcuation as indicated by Splitt and Bahrmann 1999
    lw_calc_cldy = esky * (1.0 + (0.178 - 0.00957 * (T - 290.0))) * stefan * T**4

    atts = {'long_name': 'Clear Sky Estimate-(Monteith, 1973)', 'units': 'W/m^2'}
    da = xr.DataArray(lw_calc_clear, coords={'time': ds['time'].values}, dims=['time'], attrs=atts)
    ds['monteith_clear'] = da

    atts = {'long_name': 'Overcast Sky Estimate-(Monteith, 1973)', 'units': 'W/m^2'}
    da = xr.DataArray(lw_calc_cldy, coords={'time': ds['time'].values}, dims=['time'], attrs=atts)
    ds['monteith_cloudy'] = da

    atts = {'long_name': 'Clear Sky Estimate-(Prata, 1996)', 'units': 'W/m^2'}
    da = xr.DataArray(
        lw_calc_clear_prata,
        coords={'time': ds['time'].values},
        dims=['time'],
        attrs=atts,
    )
    ds['prata_clear'] = da

    return ds
