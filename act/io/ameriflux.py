"""
This module contains I/O operations for the U.S. Department of Energy
AmeriFlux program (https://ameriflux.lbl.gov/).
"""

import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr


def read_ameriflux(
    filename,
    metadata_filename=None,
    data_type=None,
    timestep=None,
    rename_vars_dict=None,
    variable_units_dict=None,
):
    """
    Returns `xarray.Dataset` with stored data from FLUXNET or BASE-BADM and
    possible metadata from a user-defined metadata excel files for a single datastream.

    Parameters
    ----------
    filename : str
        Filename of Ameriflux dataset
    metadata_filename : str
        Excel file usually provided with Ameriflux dataset that contains the data's metadata.
        Default is None.
    data_type : str
        Type of data file to be read. Valid options are 'fluxnet' and 'base'.
        Default is None and will try to determine data type from filename.
    timestep : str
        Timestep of data, this parameter is only used for 'fluxnet' data types and if
        the time format can't be determined by the filename.
        Default is None, options are 'year', 'month', 'week', 'day', and 'hour'.
    rename_vars_dict : dict
        A dictionary containing current variable names and new variable names to replace
        the current variable names.
        Default is None and variables are not renamed.
    variable_units_dict : dict
        A dictionary containing current variable names and units to be added to that variable.
        Default is None and uses current units from Ameriflux's unit database.


    Returns
    -------
    ds : xarray.Dataset
        ACT Xarray dataset.

    """
    default_units_dict = {
        'COND_WATER': 'S cm-1',
        'DO': 'mol L-1',
        'PCH4': 'nmolCH4 mol-1',
        'PCO2': 'molCO2 mol-1',
        'PN2O': 'nmolN2O mol-1',
        'PPFD_UW_IN': 'molPhotons m-2 s-1',
        'TW': 'deg C',
        'DBH': 'cm',
        'LEAF_WET': '%',
        'SAP_DT': 'deg C',
        'SAP_FLOW': 'mmolH2O m-2 s-1',
        'T_BOLE': 'deg C',
        'T_CANOPY': 'deg C',
        'FETCH_70': 'm',
        'FETCH_80': 'm',
        'FETCH_90': 'm',
        'FETCH_FILTER': 'nondimensional',
        'FETCH_MAX': 'm',
        'CH4': 'nmolCH4 mol-1',
        'CH4_MIXING_RATIO': 'nmolCH4 mol-1',
        'CO': 'nmolCO mol-1',
        'CO2': 'molCO2 mol-1',
        'CO2_MIXING_RATIO': 'molCO2 mol-1',
        'CO2_SIGMA': 'molCO2 mol-1',
        'CO2C13': '(permil)',
        'FC': 'molCO2 m-2 s-1',
        'FCH4': 'nmolCH4 m-2 s-1',
        'FN2O': 'nmolN2O m-2 s-1',
        'FNO': 'nmolNO m-2 s-1',
        'FNO2': 'nmolNO2 m-2 s-1',
        'FO3': 'nmolO3 m-2 s-1',
        'H2O': 'mmolH2O mol-1',
        'H2O_MIXING_RATIO': 'mmolH2O mol-1',
        'H2O_SIGMA': 'mmolH2O mol-1',
        'N2O': 'nmolN2O mol-1',
        'N2O_MIXING_RATIO': 'nmolN2O mol-1',
        'NO': 'nmolNO mol-1',
        'NO2': 'nmolNO2 mol-1',
        'O3': 'nmolO3 mol-1',
        'SC': 'molCO2 m-2 s-1',
        'SCH4': 'nmolCH4 m-2 s-1',
        'SN2O': 'nmolN2O m-2 s-1',
        'SNO': 'nmolNO m-2 s-1',
        'SNO2': 'nmolNO2 m-2 s-1',
        'SO2': 'nmolSO2 mol-1',
        'SO3': 'nmolO3 m-2 s-1',
        'FH2O': 'mmolH2O m-2 s-1',
        'G': 'W m-2',
        'H': 'W m-2',
        'LE': 'W m-2',
        'SB': 'W m-2',
        'SG': 'W m-2',
        'SH': 'W m-2',
        'SLE': 'W m-2',
        'PA': 'kPa',
        'PBLH': 'm',
        'RH': '%',
        'T_SONIC': 'deg C',
        'T_SONIC_SIGMA': 'deg C',
        'TA': 'deg C',
        'VPD': 'hPa',
        'D_SNOW': 'cm',
        'P': 'mm',
        'P_RAIN': 'mm',
        'P_SNOW': 'mm',
        'RUNOFF': 'mm',
        'STEMFLOW': 'mm',
        'THROUGHFALL': 'mm',
        'ALB': '%',
        'APAR': 'molPhoton m-2 s-1',
        'EVI': 'nondimensional',
        'FAPAR': '%',
        'FIPAR': '%',
        'LW_BC_IN': 'W m-2',
        'LW_BC_OUT': 'W m-2',
        'LW_IN': 'W m-2',
        'LW_OUT': 'W m-2',
        'MCRI': 'nondimensional',
        'MTCI': 'nondimensional',
        'NDVI': 'nondimensional',
        'NETRAD': 'W m-2',
        'NIRV': 'W m-2 sr-1 nm-1',
        'PPFD_BC_IN': 'molPhoton m-2 s-1',
        'PPFD_BC_OUT': 'molPhoton m-2 s-1',
        'PPFD_DIF': 'molPhoton m-2 s-1',
        'PPFD_DIR': 'molPhoton m-2 s-1',
        'PPFD_IN': 'molPhoton m-2 s-1',
        'PPFD_OUT': 'molPhoton m-2 s-1',
        'PRI': 'nondimensional',
        'R_UVA': 'W m-2',
        'R_UVB': 'W m-2',
        'REDCI': 'nondimensional',
        'REP': 'nm',
        'SPEC_NIR_IN': 'W m-2 nm-1',
        'SPEC_NIR_OUT': 'W m-2 sr-1 nm-1',
        'SPEC_NIR_REFL': 'nondimensional',
        'SPEC_PRI_REF_IN': 'W m-2 nm-1',
        'SPEC_PRI_REF_OUT': 'W m-2 sr-1 nm-1',
        'SPEC_PRI_REF_REFL': 'nondimensional',
        'SPEC_PRI_TGT_IN': 'W m-2 nm-1',
        'SPEC_PRI_TGT_OUT': 'W m-2 sr-1 nm-1',
        'SPEC_PRI_TGT_REFL': 'nondimensional',
        'SPEC_RED_IN': 'W m-2 nm-1',
        'SPEC_RED_OUT': 'W m-2 sr-1 nm-1',
        'SPEC_RED_REFL': 'nondimensional',
        'SR': 'nondimensional',
        'SW_BC_IN': 'W m-2',
        'SW_BC_OUT': 'W m-2',
        'SW_DIF': 'W m-2',
        'SW_DIR': 'W m-2',
        'SW_IN': 'W m-2',
        'SW_OUT': 'W m-2',
        'TCARI': 'nondimensional',
        'SWC': '%',
        'SWP': 'kPa',
        'TS': 'deg C',
        'TSN': 'deg C',
        'WTD': 'm',
        'MO_LENGTH': 'm',
        'TAU': 'kg m-1 s-2',
        'U_SIGMA': 'm s-1',
        'USTAR': 'm s-1',
        'V_SIGMA': 'm s-1',
        'W_SIGMA': 'm s-1',
        'WD': 'Decimal degrees',
        'WD_SIGMA': 'decimal degree',
        'WS': 'm s-1',
        'WS_MAX': 'm s-1',
        'ZL': 'nondimensional',
        'GPP': 'molCO2 m-2 s-1',
        'NEE': 'molCO2 m-2 s-1',
        'RECO': 'molCO2 m-2 s-1',
        'FC_SSITC_TEST': 'nondimensional',
        'FCH4_SSITC_TEST': 'nondimensional',
        'FN2O_SSITC_TEST': 'nondimensional',
        'FNO_SSITC_TEST': 'nondimensional',
        'FNO2_SSITC_TEST': 'nondimensional',
        'FO3_SSITC_TEST': 'nondimensional',
        'H_SSITC_TEST': 'nondimensional',
        'LE_SSITC_TEST': 'nondimensional',
        'TAU_SSITC_TEST': 'nondimensional',
    }
    # Reader section for BASE BADM datasets
    # Differs from fluxnet as there is metadata in the first few lines
    # of the csv file.
    if data_type is not None:
        if data_type.lower() == 'base':
            file_data_type = 'base'
        elif data_type.lower() == 'fluxnet':
            file_data_type = 'fluxnet'
        else:
            raise ValueError("Must choose a valid data_type, either 'base' or 'fluxnet'")
    # Try to automatically get data type if data type isn't provided
    elif data_type is None and 'FLUXNET' in filename:
        file_data_type = 'fluxnet'
    elif data_type is None and 'BASE-BADM' in filename:
        file_data_type = 'base'
    else:
        raise ValueError(
            'Could not determine the data type from the filename '
            ' Please provide a valid data type to the data_type parameter!'
        )

    if file_data_type == 'base':
        # Grab site and version metadata
        metadata = pd.read_csv(filename, header=None, nrows=2, sep=':', index_col=0)
        site = metadata.loc['# Site']
        version = metadata.loc['# Version']

        # Files are hourly, set the time format
        _format = "%Y%m%d%H%M"

        # Read in actual data
        df = pd.read_csv(filename, skiprows=2)

        # Set time and format time to datetime64
        key = 'TIMESTAMP_START'
        df['time'] = pd.to_datetime(df[key], format=_format)
        df = df.set_index('time')
        df = df.drop(columns=['TIMESTAMP_START', 'TIMESTAMP_END'])

        # Convert dataframe to xarray dataset
        ds = xr.Dataset.from_dataframe(df)

        # Add site and version to attributes
        ds.attrs['site'] = site.values[0].strip()
        ds.attrs['version'] = version.values[0].strip()

    # Reader for fluxnet files
    # Fluxnet files can be formatted in different time samplings
    elif file_data_type == 'fluxnet':
        # Checks timestep in filename, if not, will check timestep parameter
        if 'YY' in filename or timestep == 'year':
            _format = "%Y"
        elif 'MM' in filename or timestep == 'month':
            _format = "%Y%m"
        elif 'DD' in filename or 'WW' in filename or timestep == 'week' or timestep == 'day':
            _format = "%Y%m%d"
        elif 'HH' in filename or timestep == 'day':
            _format = "%Y%m%d%H%M"
        else:
            raise ValueError(
                "Incorrect timestep provided or no timestep determined from filename, "
                "please provide either year, month, week, day or hour for the timestep parameter."
            )
        # Read data into a pandas dataframe
        df = pd.read_csv(filename)

        # Set time and convert to datetime64
        if 'TIMESTAMP_START' in df:
            key = 'TIMESTAMP_START'
            df['time'] = pd.to_datetime(df[key], format=_format)
            df = df.set_index('time')
            df = df.drop(columns=['TIMESTAMP_START', 'TIMESTAMP_END'])
        else:
            key = 'TIMESTAMP'
            df['time'] = pd.to_datetime(df[key], format=_format)
            df = df.set_index('time')
            df = df.drop(columns=[key])

        # Convert to xarray dataset
        ds = xr.Dataset.from_dataframe(df)

    # Renames variables if user provides new names
    if rename_vars_dict is not None:
        ds = ds.rename_vars(rename_vars_dict)

    # Add _FILL_VALUE attribute
    for key in ds.variables.keys():
        if -9999.0 in ds.variables[key].values or -9999 in ds.variables[key].values:
            ds.variables[key].attrs['_FILL_VALUE'] = -9999

    # Add nans where fill value is present
    ds = ds.where(ds != -9999.0)

    # Add units from the unit dictionary
    # Matches keys that have different levels as well such as SWC_1_1_1
    # Only works for base dataset
    if variable_units_dict is None and file_data_type == 'base':
        for key in ds.variables.keys():
            try:
                if re.match(r'TS_[\d]', key):
                    unit_key = 'TS'
                    ds.variables[key].attrs['units'] = default_units_dict[unit_key]
                elif re.match(r'SWC_[\d]', key):
                    unit_key = 'SWC'
                    ds.variables[key].attrs['units'] = default_units_dict[unit_key]
                elif re.match(r'G_[\d]', key):
                    unit_key = 'G'
                    ds.variables[key].attrs['units'] = default_units_dict[unit_key]
                elif re.match(r'VPD_', key):
                    unit_key = 'VPD'
                    ds.variables[key].attrs['units'] = default_units_dict[unit_key]
                else:
                    ds.variables[key].attrs['units'] = default_units_dict[key]
            except KeyError:
                continue

    if variable_units_dict is not None:
        for key in ds.variables.keys():
            ds.variables[key].attrs['units'] = variable_units_dict[key]

    # Adds metadata to the dataset if a metadata file is provided
    if metadata_filename is not None:
        ds = _ameriflux_metadata_processing(ds, metadata_filename)

    return ds


def _ameriflux_metadata_processing(ds, metadata_filename):
    """Adds metadata to an ameriflux dataset if a metadata file is provided."""
    # Read in metadata excel file if provided.
    meta_df = pd.read_excel(metadata_filename)

    # Create a list for attributes and their respective values
    meta_key_list = meta_df['VARIABLE'].to_list()
    meta_value_list = meta_df['DATAVALUE'].to_list()

    # Add attrs that are non duplicates as duplicates require more processing below
    non_duplicates = [item for item in meta_key_list if meta_key_list.count(item) == 1]
    data_dict = defaultdict(list)
    for i, j in zip(meta_key_list, meta_value_list):
        data_dict[i].append(j)
    for key in non_duplicates:
        ds.attrs[key] = data_dict[key][0]

    # The code below to retrieve group metadata was done so that
    # If the order changes of these attributes, which it can, it will find them regardless
    # Also checks if a specific attribute is missing for a team member but not others
    meta_arr = np.array(meta_key_list)
    # Retrieve team_member metadata
    team_indices = np.where(meta_arr == 'TEAM_MEMBER_NAME')[0]
    team_members = []
    valid_names = [
        'TEAM_MEMBER_NAME',
        'TEAM_MEMBER_EMAIL',
        'TEAM_MEMBER_INSTITUTION',
        'TEAM_MEMBER_ROLE',
        'TEAM_MEMBER_ORCID',
        'TEAM_MEMBER_ADDRESS',
    ]
    for i, j in enumerate(team_indices):
        if j == team_indices[-1]:
            team_info_range = np.arange(team_indices[i], team_indices[i] + 5)
            team_members_str = []
            [
                team_members_str.append(str(meta_key_list[k]) + ':' + str(meta_value_list[k]))
                for k in team_info_range
                if meta_key_list[k] in valid_names
            ]
            result_str = ", ".join(team_members_str)
            team_members.append(result_str)
            break
        else:
            team_info_range = np.arange(team_indices[i], team_indices[i + 1])
            team_members_str = []
            [
                team_members_str.append(str(meta_key_list[k]) + ':' + str(meta_value_list[k]))
                for k in team_info_range
                if meta_key_list[k] in valid_names
            ]
            result_str = ", ".join(team_members_str)
            team_members.append(result_str)
    ds.attrs['TEAM_MEMBERS'] = team_members

    # Retrieve doi contributor metadata
    doi_indices = np.where(meta_arr == 'DOI_CONTRIBUTOR_DATAPRODUCT')[0]
    doi_members = []
    valid_doi_names = [
        'DOI_CONTRIBUTOR_DATAPRODUCT',
        'DOI_CONTRIBUTOR_NAME',
        'DOI_CONTRIBUTOR_ROLE',
        'DOI_CONTRIBUTOR_INSTITUTION',
        'DOI_CONTRIBUTOR_EMAIL',
        'DOI_CONTRIBUTOR_ORCID',
        'DOI_CONTRIBUTOR_DATE_START',
        'DOI_CONTRIBUTOR_DATE_END',
        'DOI CONTRIBUTOR_ADDRESS',
    ]
    for i, j in enumerate(doi_indices):
        if j == doi_indices[-1]:
            doi_info_range = np.arange(doi_indices[i], doi_indices[i] + 5)
            doi_contrib_str = []
            [
                doi_contrib_str.append(str(meta_key_list[k]) + ':' + str(meta_value_list[k]))
                for k in doi_info_range
                if meta_key_list[k] in valid_doi_names
            ]
            result_doi_str = ", ".join(doi_contrib_str)
            doi_members.append(result_doi_str)
            break
        else:
            doi_info_range = np.arange(doi_indices[i], doi_indices[i + 1])
            doi_contrib_str = []
            [
                doi_contrib_str.append(str(meta_key_list[k]) + ':' + str(meta_value_list[k]))
                for k in doi_info_range
                if meta_key_list[k] in valid_doi_names
            ]
            result_doi_str = ", ".join(doi_contrib_str)
            doi_members.append(result_doi_str)
    ds.attrs['DOI_CONTRIBUTORS'] = doi_members

    # Retrieve flux method metadata
    flux_indices = np.where(meta_arr == 'FLUX_MEASUREMENTS_METHOD')[0]
    flux_members = []
    valid_flux_names = [
        'FLUX_MEASUREMENTS_METHOD',
        'FLUX_MEASUREMENTS_VARIABLE',
        'FLUX_MEASUREMENTS_DATE_START',
        'FLUX_MEASUREMENTS_DATE_END',
        'FLUX_MEASUREMENTS_OPERATIONS',
    ]
    for i, j in enumerate(flux_indices):
        if j == flux_indices[-1]:
            flux_info_range = np.arange(flux_indices[i], flux_indices[i] + 5)
            flux_str = []
            [
                flux_str.append(str(meta_key_list[k]) + ':' + str(meta_value_list[k]))
                for k in flux_info_range
                if meta_key_list[k] in valid_flux_names
            ]
            result_flux_str = ", ".join(flux_str)
            flux_members.append(result_flux_str)
            break
        else:
            flux_info_range = np.arange(flux_indices[i], flux_indices[i + 1])
            flux_str = []
            [
                flux_str.append(str(meta_key_list[k]) + ':' + str(meta_value_list[k]))
                for k in flux_info_range
                if meta_key_list[k] in valid_flux_names
            ]
            result_flux_str = ", ".join(flux_str)
            flux_members.append(result_flux_str)
    ds.attrs['FLUX_MEASUREMENTS_METHODS'] = flux_members
    return ds


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
