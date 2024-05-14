"""
This module contains I/O operations for the U.S. Department of Energy
AmeriFlux program (https://ameriflux.lbl.gov/).
"""

import numpy as np
import pandas as pd
import warnings


def convert_to_ameriflux(
    ds,
    variable_mapping=None,
    soil_mapping=None,
    depth_profile=[2.5, 5, 10, 15, 20, 30, 35, 50, 75, 100],
    include_missing_variables=False,
    **kwargs,
):
    """

    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ARM-standard netCDF files from a single datastream. Has some procedures
    to ensure time is correctly fomatted in returned Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of data to convert to AmeriFlux format
    variable_mapping : dict
        Dictionary of variables mappings.  The key should be the name of the variable
        in the Dataset with the values being dictionaries of the AmeriFlux name and units.
        For example:
            var_mapping = {
                'co2_flux': {'name': 'FC', 'units': 'umol/(m^2 s)'},
            }
    soil_mapping : dict
        Dictionary of soil variables mappings following the same formatting as variable_mapping.
        It is understood that the AmeriFlux name may be the same for some variables.  This
        script attempts to automatically name these measurements. If a variable is not dimensioned
        by a depth nor has a sensor_height attribute, it will automatically assume that it's
        at the first depth in the depth_profile variable.
    depth_profile : list
        List of depths that the variables will be mapped to.  If a depth is not in this list,
        the index chosen will be the one closest to the depth value.
    include_missing_variables : boolean
        If there variables that are completely missing (-9999) chose whether or not to include
        them in the DataFrame.

    Returns
    -------
    df : pandas.DataFrame (or None)
        Returns a pandas dataframe for easy writing to csv

    """
    # Use ARM variable mappings if none provided
    if variable_mapping is None:
        warnings.warn('Variable mapping was not provided, using default ARM mapping')
        # Define variable mapping and units
        # The key is the variable name in the data and the name in the dictionary
        # is the AmeriFlux Name
        var_mapping = {
            'co2_flux': {'name': 'FC', 'units': 'umol/(m^2 s)'},
            'co2_molar_fraction': {'name': 'CO2', 'units': 'nmol/mol'},
            'co2_mixing_ratio': {'name': 'CO2_MIXING_RATIO', 'units': 'umol/mol'},
            'h2o_mole_fraction': {'name': 'H2O', 'units': 'mmol/mol'},
            'h2o_mixing_ratio': {'name': 'H2O_MIXING_RATIO', 'units': 'mmol/mol'},
            'ch4_mole_fraction': {'name': 'CH4', 'units': 'nmol/mol'},
            'ch4_mixing_ratio': {'name': 'CH4_MIXING_RATIO', 'units': 'nmol/mol'},
            'momentum_flux': {'name': 'TAU', 'units': 'kg/(m s^2)'},
            'sensible_heat_flux': {'name': 'H', 'units': 'W/m^2'},
            'latent_flux': {'name': 'LE', 'units': 'W/m^2'},
            'air_temperature': {'name': 'TA', 'units': 'deg C'},
            'air_pressure': {'name': 'PA', 'units': 'kPa'},
            'relative_humidity': {'name': 'RH', 'units': '%'},
            'sonic_temperature': {'name': 'T_SONIC', 'units': 'deg C'},
            'water_vapor_pressure_defecit': {'name': 'VPD', 'units': 'hPa'},
            'Monin_Obukhov_length': {'name': 'MO_LENGTH', 'units': 'm'},
            'Monin_Obukhov_stability_parameter': {'name': 'ZL', 'units': ''},
            'mean_wind': {'name': 'WS', 'units': 'm/s'},
            'wind_direction_from_north': {'name': 'WD', 'units': 'deg'},
            'friction_velocity': {'name': 'USTAR', 'units': 'm/s'},
            'maximum_instantaneous_wind_speed': {'name': 'WS_MAX', 'units': 'm/s'},
            'down_short_hemisp': {'name': 'SW_IN', 'units': 'W/m^2'},
            'up_short_hemisp': {'name': 'SW_OUT', 'units': 'W/m^2'},
            'down_long': {'name': 'LW_IN', 'units': 'W/m^2'},
            'up_long': {'name': 'LW_OUT', 'units': 'W/m^2'},
            'albedo': {'name': 'ALB', 'units': '%'},
            'net_radiation': {'name': 'NETRAD', 'units': 'W/m^2'},
            'par_inc': {'name': 'PPFD_IN', 'units': 'umol/(m^2 s)'},
            'par_ref': {'name': 'PPFD_OUT', 'units': 'umol/(m^2 s)'},
            'precip': {'name': 'P', 'units': 'mm'},
        }

    # Use ARM variable mappings if none provided
    # Similar to the above.  This has only been tested on the ARM
    # ECOR, SEBS, STAMP, and AMC combined.  The automated naming may
    # not work for all cases
    if soil_mapping is None:
        warnings.warn('Soil variable mapping was not provided, using default ARM mapping')
        soil_mapping = {
            'surface_soil_heat_flux': {'name': 'G', 'units': 'W/m^2'},
            'soil_temp': {'name': 'TS', 'units': 'deg C'},
            'temp': {'name': 'TS', 'units': 'deg C'},
            'soil_moisture': {'name': 'SWC', 'units': '%'},
            'soil_specific_water_content': {'name': 'SWC', 'units': '%'},
            'vwc': {'name': 'SWC', 'units': '%'},
        }

    # Loop through variables and update units to the AmeriFlux standard
    for v in ds:
        if v in var_mapping:
            ds = ds.utils.change_units(variables=v, desired_unit=var_mapping[v]['units'])

    # Get start/end time stamps
    ts_start = ds['time'].dt.strftime('%Y%m%d%H%M').values
    ts_end = [
        pd.to_datetime(t + np.timedelta64(30, 'm')).strftime('%Y%m%d%H%M')
        for t in ds['time'].values
    ]
    data = {}
    data['TIMESTAMP_START'] = ts_start
    data['TIMESTAMP_END'] = ts_end

    # Loop through the variables in the var mapping dictionary and add data to dictionary
    for v in var_mapping:
        if v in ds:
            if 'missing_value' not in ds[v].attrs:
                ds[v].attrs['missing_value'] = -9999
            if np.all(ds[v].isnull()):
                if include_missing_variables:
                    data[var_mapping[v]['name']] = ds[v].values
            else:
                data[var_mapping[v]['name']] = ds[v].values
        else:
            if include_missing_variables:
                data[var_mapping[v]['name']] = np.full(ds['time'].shape, -9999)

    # Automated naming for the soil variables
    # Again, this may not work for other cases.  Careful review is needed.
    prev_var = ''
    for var in soil_mapping:
        if soil_mapping[var]['name'] != prev_var:
            h = 1
            r = 1
            prev_var = soil_mapping[var]['name']
        soil_vars = [
            v2
            for v2 in list(ds)
            if (v2.startswith(var)) & ('std' not in v2) & ('qc' not in v2) & ('net' not in v2)
        ]
        for i, svar in enumerate(soil_vars):
            vert = 1
            if ('avg' in svar) | ('average' in svar):
                continue
            soil_data = ds[svar].values
            data_shape = soil_data.shape
            if len(data_shape) > 1:
                coords = ds[svar].coords
                depth_name = list(coords)[-1]
                depth_values = ds[depth_name].values
                for depth_ind in range(len(depth_values)):
                    soil_data_depth = soil_data[:, depth_ind]
                    vert = np.where(depth_profile == depth_values[depth_ind])[0][0] + 1
                    new_name = '_'.join([soil_mapping[var]['name'], str(h), str(vert), str(r)])
                    data[new_name] = soil_data_depth
            else:
                if 'sensor_height' in ds[svar].attrs:
                    sensor_ht = ds[svar].attrs['sensor_height'].split(' ')
                    depth = abs(float(sensor_ht[0]))
                    units = sensor_ht[1]
                    if units == 'cm':
                        vert = np.argmin(np.abs(np.array(depth_profile) - depth)) + 1
                new_name = '_'.join([soil_mapping[var]['name'], str(h), str(vert), str(r)])
                data[new_name] = soil_data
            h += 1

    # Convert dictionary to dataframe and return
    df = pd.DataFrame(data)
    df = df.fillna(-9999.0)

    return df
