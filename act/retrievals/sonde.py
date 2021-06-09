"""
Functions for radiosonde related calculations.

"""

import warnings

import numpy as np

try:
    from pkg_resources import DistributionNotFound
    import metpy.calc as mpcalc
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False
except (ModuleNotFoundError, DistributionNotFound):
    warnings.warn("MetPy is installed but could not be imported. " +
                  "Please check your MetPy installation. Some features " +
                  "will be disabled.", ImportWarning)
    METPY_AVAILABLE = False

if METPY_AVAILABLE:
    from metpy.units import units


def calculate_precipitable_water(ds, temp_name='tdry', rh_name='rh',
                                 pres_name='pres'):
    """

    Function to calculate precipitable water vapor from ARM sondewnpn b1 data.
    Will first calculate saturation vapor pressure of all data using Arden-Buck
    equations, then calculate specific humidity and integrate over all pressure
    levels to give us a precipitable water value in centimeters.

    ds : ACT object
        Object as read in by the ACT netCDF reader.
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
            sat_vap_pres.append(0.61121 * np.exp((18.678 - (t / 234.5)) *
                                (t / (257.14 + t))))
        # Over ice, below freezing
        else:
            sat_vap_pres.append(0.61115 * np.exp((23.036 - (t / 333.7)) *
                                (t / (279.82 + t))))

    # convert rh from % to decimal
    rel_hum = []
    for r in np.nditer(rh):
        rel_hum.append(r / 100.)

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
        pwv = pwv + 0.5 * (spec_hum[i] + spec_hum[i - 1]) * (pressure[i - 1] -
                                                             pressure[i])

    pwv = pwv / 0.098
    return pwv


def calculate_stability_indicies(ds, temp_name="temperature",
                                 td_name="dewpoint_temperature",
                                 p_name="pressure",
                                 moving_ave_window=0):
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
    if not METPY_AVAILABLE:
        raise ImportError("MetPy need to be installed on your system to " +
                          "calculate stability indices")

    t = ds[temp_name]
    td = ds[td_name]
    p = ds[p_name]

    if not hasattr(t, "units"):
        raise AttributeError("Temperature field must have units" +
                             " for ACT to discern!")

    if not hasattr(td, "units"):
        raise AttributeError("Dewpoint field must have units" +
                             " for ACT to discern!")

    if not hasattr(p, "units"):
        raise AttributeError("Pressure field must have units" +
                             " for ACT to discern!")
    if t.units == "C":
        t_units = units.degC
    else:
        t_units = getattr(units, t.units)

    if td.units == "C":
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
        t_sorted = np.convolve(
            t_sorted, np.ones((moving_ave_window,)) / moving_ave_window)
        td_sorted = np.convolve(
            td_sorted, np.ones((moving_ave_window,)) / moving_ave_window)
        p_sorted = np.convolve(
            p_sorted, np.ones((moving_ave_window,)) / moving_ave_window)

    t_sorted = t_sorted * t_units
    td_sorted = td_sorted * td_units
    p_sorted = p_sorted * p_units

    t_profile = mpcalc.parcel_profile(
        p_sorted, t_sorted[0], td_sorted[0])

    # Calculate parcel trajectory
    ds["parcel_temperature"] = t_profile.magnitude
    ds["parcel_temperature"].attrs['units'] = t_profile.units

    # Calculate CAPE, CIN, LCL
    sbcape, sbcin = mpcalc.surface_based_cape_cin(
        p_sorted, t_sorted, td_sorted)
    lcl = mpcalc.lcl(
        p_sorted[0], t_sorted[0], td_sorted[0])
    try:
        lfc = mpcalc.lfc(
            p_sorted[0], t_sorted[0], td_sorted[0])
    except IndexError:
        lfc = np.nan * p_sorted.units

    mucape, mucin = mpcalc.most_unstable_cape_cin(
        p_sorted, t_sorted, td_sorted)

    where_500 = np.argmin(np.abs(p_sorted - 500 * units.hPa))
    li = t_sorted[where_500] - t_profile[where_500]

    ds["surface_based_cape"] = sbcape.magnitude
    ds["surface_based_cape"].attrs['units'] = "J/kg"
    ds["surface_based_cape"].attrs['long_name'] = "Surface-based CAPE"
    ds["surface_based_cin"] = sbcin.magnitude
    ds["surface_based_cin"].attrs['units'] = "J/kg"
    ds["surface_based_cin"].attrs['long_name'] = "Surface-based CIN"
    ds["most_unstable_cape"] = mucape.magnitude
    ds["most_unstable_cape"].attrs['units'] = "J/kg"
    ds["most_unstable_cape"].attrs['long_name'] = "Most unstable CAPE"
    ds["most_unstable_cin"] = mucin.magnitude
    ds["most_unstable_cin"].attrs['units'] = "J/kg"
    ds["most_unstable_cin"].attrs['long_name'] = "Most unstable CIN"
    ds["lifted_index"] = li.magnitude
    ds["lifted_index"].attrs['units'] = t_profile.units
    ds["lifted_index"].attrs['long_name'] = "Lifted index"
    ds["level_of_free_convection"] = lfc.magnitude
    ds["level_of_free_convection"].attrs['units'] = lfc.units
    ds["level_of_free_convection"].attrs['long_name'] = "Level of free convection"
    ds["lifted_condensation_level_temperature"] = lcl[1].magnitude
    ds["lifted_condensation_level_temperature"].attrs['units'] = lcl[1].units
    ds["lifted_condensation_level_temperature"].attrs['long_name'] = "Lifted condensation level temperature"
    ds["lifted_condensation_level_pressure"] = lcl[0].magnitude
    ds["lifted_condensation_level_pressure"].attrs['units'] = lcl[0].units
    ds["lifted_condensation_level_pressure"].attrs['long_name'] = "Lifted condensation level pressure"
    return ds
