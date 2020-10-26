
""" Calculating SIRS Radiation Estimates """

import numpy as np
from astral import sun
from scipy.constants import Stefan_Boltzmann
from datetime import datetime, timedelta
import astral
from act.utils.datetime_utils import datetime64_to_datetime


def calculate_sirs_variable(sirs_obj, met_obj, sirs_time='time', met_time='time', lat='lat', lon='lon',
                             downwelling_sw_diffuse_hemisp_irradiance='down_short_diffuse_hemisp',
                             shortwave_direct_normal_irradiance='short_direct_normal',
                             downwelling_sw_hemisp_irradiance='down_short_hemisp',
                             mean_temperature='temp_mean', calculated_mean_vapor_pressure='vapor_pressure_mean'):

    """

    Functions to calculate various SIRS irradiance measurements. Returns a new
    SIRS object with calculations included as new variables.

    Parameters
    ----------
    sirs_obj : ACT object
        SIRS object as read in by the ACT netCDF reader.
    met_obj : ACT object
        MET object as read in by the ACT netCDF reader. Must be the MET sensor
        closest to the SIRS instrument of sirs_obj
    sirs_time : str
        Name of SIRS time field to use. Defaults to 'time'.
    met_time : str
        Name of MET time field to use. Defaults to 'time'.
    lat : str
        Name of SIRS lat field to use. Defaults to 'lat'.
    lon : str
        Name of SIRS lon field to use. Defaults to 'lon'.
    downwelling_sw_diffuse_hemisp_irradiance : str
        Name of SIRS downwelling shortwave diffuse hemispheric irradiance field to use.
        Defaults to 'down_short_diffuse_hemisp'.
    shortwave_direct_normal_irradiance : str
        Name of SIRS shortwave direct normal hemispheric irradiance field to use.
        Defaults to 'short_direct_normal'.
    downwelling_sw_hemisp_irradiance : str
        Name of SIRS downwelling shortwave hemispheric irradiance field to use.
        Defaults to 'down_short_hemisp'.
    mean_temperature : str
        Name of MET mean temperature field to use.
        Defaults to 'temp_mean'.
    calculated_mean_vapor_pressure : str
        Name of MET mean vapor pressure field to use.
        Defaults to 'vapor_pressure_mean'.


    Returns
    -------

    sirs_obj: ACT Object
        The new SIRS object with calculations included as new variables.

    """

    time = sirs_obj[sirs_time].values
    lat = sirs_obj[lat]
    lon = sirs_obj[lon]

    # -------------------------------------
    # Calculating Derived Down Short Hemisp
    # -------------------------------------

    solar_zenith = np.full(len(time), np.nan)
    obs = astral.Observer(latitude=lat, longitude=lon)
    tt = datetime64_to_datetime(time)
    for ii, tm in enumerate(time):
        solar_zenith[ii] = np.cos(np.radians(sun.zenith(obs, tt[ii])))

    derived_h = (sirs_obj[downwelling_sw_diffuse_hemisp_irradiance] +
                 (solar_zenith * sirs_obj[shortwave_direct_normal_irradiance]))

    sirs_obj['derived_down_short_hemisp'] = derived_h
    sirs_obj['derived_down_short_hemisp'].attrs['long_name'] = 'Derived Down Shortwave Hemispheric'
    sirs_obj['derived_down_short_hemisp'].attrs['units'] = 'W/m^2'

    # ---------------------------------
    # Calculating Irradiance Difference
    # ---------------------------------
    diff_sh = derived_h - sirs_obj[downwelling_sw_hemisp_irradiance]
    sirs_obj['diff_short_hemisp'] = diff_sh
    sirs_obj['diff_short_hemisp'].attrs['long_name'] = 'Irradiance Difference'
    sirs_obj['diff_short_hemisp'].attrs['units'] = 'W/m^2'

    # ---------------------------------
    # Calculating Irradiance Ratio
    # ---------------------------------
    ratio_sh = derived_h / sirs_obj[downwelling_sw_hemisp_irradiance]
    index = np.where((derived_h < 60.) & (sirs_obj[downwelling_sw_hemisp_irradiance] < 60.))
    ratio_sh.load()[index] = np.nan
    sirs_obj['ratio_short_hemisp'] = ratio_sh
    sirs_obj['ratio_short_hemisp'].attrs['long_name'] = 'Irradiance Ratio'
    sirs_obj['ratio_short_hemisp'].attrs['units'] = ''

    # ---------------------------------
    # Calculating Longwave Stuff
    # ---------------------------------

    if met_obj is not None:

        derived_time = met_obj[met_time]
        T = met_obj[mean_temperature] + 273.15  # C to K
        P = met_obj[calculated_mean_vapor_pressure] * 10.  # kpa to hpa

        # Perform calculations
        stefan = Stefan_Boltzmann
        esky = 0.61 + 0.06 * np.sqrt(P)
        lw_calc_clear = esky * stefan * T**4
        xi = 46.5 * (P / T)
        lw_calc_clear_new = (1.0 - (1.0 + xi) * np.exp(-(1.2 + 3.0 * xi)**.5)) * stefan * T**4
        lw_calc_cldy = esky * (1.0 + (0.178 - 0.00957 * (T - 290.))) * stefan * T**4

        sirs_obj['montieth_clear'] = lw_calc_clear
        sirs_obj['montieth_cloud'] = lw_calc_cldy
        sirs_obj['prata_clear'] = lw_calc_clear_new
        sirs_obj['derived_time'] = derived_time

    else:
        nan_data = np.full(len(time), np.nan)
        sirs_obj['montieth_clear'] = nan_data
        sirs_obj['montieth_cloud'] = nan_data
        sirs_obj['prata_clear'] = nan_data
        sirs_obj['derived_time'] = nan_data

    sirs_obj['montieth_clear'].attrs['long_name'] = 'Clear Sky Estimate-(Montieth, 1973)'
    sirs_obj['montieth_clear'].attrs['units'] = 'W/m^2'

    sirs_obj['montieth_cloud'].attrs['long_name'] = 'Overcast Sky Estimate-(Montieth, 1973)'
    sirs_obj['montieth_cloud'].attrs['units'] = 'W/m^2'

    sirs_obj['prata_clear'].attrs['long_name'] = 'Clear Sky Estimate-(Prata, 1996)'
    sirs_obj['prata_clear'].attrs['units'] = 'W/m^2'

    sirs_obj['derived_time'].attrs['long_name'] = 'Time from Met Object'
    sirs_obj['derived_time'].attrs['units'] = 'W/m^2'

    # Calculate Net Radiation
    ush = sirs_obj['up_short_hemisp']
    ulh = sirs_obj['up_long_hemisp']
    dsh = sirs_obj['down_short_hemisp']
    dlh = sirs_obj['down_long_hemisp_shaded']

    sirs_net = -ush + dsh - ulh + dlh

    try:
        sirs_net = sirs_net.rolling(time=30).mean()
        sirs_obj['net_radiation'] = sirs_net
        sirs_obj['net_radiation'].attrs['long_name'] = 'DQO Calculated Net Radiation - Moving Average of 30'
        sirs_obj['net_radiation'].attrs['units'] = 'W/m^2'
    except Exception:
        sirs_obj['net_radiation'] = sirs_net
        sirs_obj['net_radiation'].attrs['long_name'] = 'DQO Calculated Net Radiation'
        sirs_obj['net_radiation'].attrs['units'] = 'W/m^2'

    return sirs_obj
