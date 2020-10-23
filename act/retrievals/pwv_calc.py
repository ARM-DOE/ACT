""" Retrievals for precipitable water vapor. """

import numpy as np


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
