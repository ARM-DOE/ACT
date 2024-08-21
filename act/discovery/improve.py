"""
Script for downloading data from the IMPROVE network

"""

import pandas as pd
import numpy as np
import xarray as xr


def get_improve_data(site_id=None, parameter_id=None, start_date=None, end_date=None):
    """
    Retrieve IMPROVE data for the given site and variable ids and store it in an
    xarray dataset. Documentation on the IMPROVE data can be found at
    https://vista.cira.colostate.edu/Improve/data-user-guide/

    Also adds in metadata from the site summary page to the global attributes
    https://views.cira.colostate.edu/adms/Pub/SiteSummary.aspx?dsidse=10001&siidse=244

    Parameters
    ----------
    site_id : str
        Site id number which can be retrieved from the IMPROVE page for each site such as
        https://views.cira.colostate.edu/adms/Pub/SiteSummary.aspx?dsidse=10001&siidse=244
    parameter_id : list
        List of parameter id values to retrieve from the API.
    start_date : str
        Start date formatted as M/D/YEAR such as 1/31/2022
    end_date : str
        End date formatted as M/D/YEAR such as 1/31/2022

    Returns
    -------
    ds : xarray.Dataset
        Returns an Xarray dataset object

    Example
    -------
    act.discovery.get_improve_data(site_id='244')

    """

    # Build URL
    base_url = 'https://views.cira.colostate.edu/fed/svc/DataSvc.aspx?action=getqueryresults&cmdfileid=ServiceSqlCommandFile&cmdid=BasicDataQuery1_Codes'

    if site_id is None:
        raise ValueError('Please provide a site_id')
    else:
        base_url += '&dsidse=10001&siidse=' + str(site_id)

    if parameter_id is None:
        base_url += '&paidse=101,136,907,900,102,104,105,115,116,117,114,3778,142,143,144,145,3016,146,3699,141,3779,3217,108,109,112,113,301,304,303,3716,3717,3718,3719,3720,3721,3722,3730,3731,3732,3733,3734,3735,3736,3694,121,3723,3724,3725,3726,3727,3728,3729,3737,3738,3739,3740,3741,3742,3743,118,148,128,130,132,941,127,903,910,3744,3745,3746,3747,3748,3749,3750,3751,3752,3753,3754,3755,3756,3757,131,138,139,133,3704,3705,3706,3707,3708,3709,3710,3711,3712,3713,3714,3715,147,124,150,3695,3014,153,154,134,911,158,156,151,202,159,160,162,163'
    else:
        base_url += '&paidse=' + ','.join(parameter_id)

    if start_date is None:
        raise ValueError('Please provide a start date')
    else:
        base_url += '&sd=' + start_date
    if end_date is None:
        raise ValueError('Please provide an end date')
    else:
        base_url += '&ed=' + end_date

    # Read data and get variables
    df = pd.read_html(base_url)[0]
    variables = np.unique(df.Param)

    # Print out proper acknowledgement
    print("Please use the following acknowledgment when using IMPROVE data:\n")

    print(
        "IMPROVE is a collaborative association of state, tribal, and federal agencies, and international partners. US Environmental Protection Agency is the primary funding source, with contracting and research support from the National Park Service. The Air Quality Group at the University of California, Davis is the central analytical laboratory, with ion analysis provided by Research Triangle Institute, and carbon analysis provided by Desert Research Institute."
    )

    # Creat mapping of variable names to metadata
    mapping = {
        'ALf': {'name': 'aluminum_fine', 'long_name': 'Aluminum (Fine)', 'epa_code': '88104'},
        'ASf': {'name': 'arsenic_fine', 'long_name': 'Arsenic (Fine)', 'epa_code': '88103'},
        'BRf': {'name': 'bromine_fine', 'long_name': 'Bromine (Fine)', 'epa_code': '88109'},
        'CAf': {'name': 'calcium_fine', 'long_name': 'Calcium (Fine)', 'epa_code': '88111'},
        'CLf': {'name': 'chlorine_fine', 'long_name': 'Chlorine (Fine)', 'epa_code': '88115'},
        'CRf': {'name': 'chromium_fine', 'long_name': 'Chromium (Fine)', 'epa_code': '88112'},
        'CUf': {'name': 'copper_fine', 'long_name': 'Copper (Fine)', 'epa_code': '88114'},
        'FEf': {'name': 'iron_fine', 'long_name': 'Iron (Fine)', 'epa_code': '88126'},
        'PBf': {'name': 'lead_fine', 'long_name': 'Lead (Fine)', 'epa_code': '88128'},
        'MGf': {'name': 'magnesium_fine', 'long_name': 'Magnesium (Fine)', 'epa_code': '88140'},
        'MNf': {'name': 'manganese_fine', 'long_name': 'Manganese (Fine)', 'epa_code': '88132'},
        'NIf': {'name': 'nickel_fine', 'long_name': 'Nickel (Fine)', 'epa_code': '88136'},
        'Pf': {'name': 'phosphorus_fine', 'long_name': 'Phosphorus (Fine)', 'epa_code': '88152'},
        'Kf': {'name': 'potassium_fine', 'long_name': 'Potassium (Fine)', 'epa_code': '88180'},
        'RBf': {'name': 'rubidium_fine', 'long_name': 'Rubidium (Fine)', 'epa_code': '88176'},
        'SEf': {'name': 'selenium_fine', 'long_name': 'Selenium (Fine)', 'epa_code': '88154'},
        'SIf': {'name': 'silicon_fine', 'long_name': 'Silicon (Fine)', 'epa_code': '88165'},
        'NAf': {'name': 'sodium_fine', 'long_name': 'Sodium (Fine)', 'epa_code': '88184'},
        'SRf': {'name': 'strontium_fine', 'long_name': 'Strontium (Fine)', 'epa_code': '88168'},
        'Sf': {'name': 'sulfur_fine', 'long_name': 'Sulfur (Fine)', 'epa_code': '88169'},
        'TIf': {'name': 'titanium_fine', 'long_name': 'Titanium (Fine)', 'epa_code': '88161'},
        'Vf': {'name': 'vanadium_fine', 'long_name': 'Vanadium (Fine)', 'epa_code': '88164'},
        'ZNf': {'name': 'zinc_fine', 'long_name': 'Zinc (Fine)', 'epa_code': '88167'},
        'ZRf': {'name': 'zirconium_fine', 'long_name': 'Zirconium (Fine)', 'epa_code': '88185'},
        'CHLf': {'name': 'chloride_fine', 'long_name': 'Chloride (Fine)', 'epa_code': '88203'},
        'NO3f': {'name': 'nitrate_fine', 'long_name': 'Nitrate (Fine)', 'epa_code': '88306'},
        'N2f': {'name': 'nitrite_fine', 'long_name': 'Nitrite (Fine)', 'epa_code': '88338'},
        'SO4f': {'name': 'sulfate_fine', 'long_name': 'Sulfate (Fine)', 'epa_code': '88403'},
        'OC1f': {
            'name': 'carbon_organic_fraction_1_fine',
            'long_name': 'Carbon, Organic Fraction 1 (Fine)',
            'comments': 'TOR, pure helium (>99.999%) atmosphere, temperature (T) = 140 °C',
            'epa_code': '88324',
        },
        'OC2f': {
            'name': 'carbon_organic_fraction_2_fine',
            'long_name': 'Carbon, Organic Fraction 2 (Fine)',
            'comments': 'TOR, pure helium (>99.999%) atmosphere, temperature (T) = 280 °C',
            'epa_code': '88325',
        },
        'OC3f': {
            'name': 'carbon_organic_fraction_3_fine',
            'long_name': 'Carbon, Organic Fraction 3 (Fine)',
            'comments': 'TOR, pure helium (>99.999%) atmosphere, temperature (T) = 480 °C',
            'epa_code': '88326',
        },
        'OC4f': {
            'name': 'carbon_organic_fraction_4_fine',
            'long_name': 'Carbon, Organic Fraction 4 (Fine)',
            'comments': 'TOR, pure helium (>99.999%) atmosphere, temperature (T) = 580 °C',
            'epa_code': '88327',
        },
        'OPf': {
            'name': 'carbon_organic_reflectance_fine',
            'long_name': 'Carbon, Organic Pyrolized (Fine) by Reflectance',
            'comments': 'TOR, carbon that is measured after the introduction of helium/oxygen atmosphere at °550 C but beforereflectance returns to initial value',
            'epa_code': '88328',
        },
        'OPTf': {
            'name': 'carbon_organic_transmittance_fine',
            'long_name': 'Carbon, Organic Pyrolized (Fine) by Transmittance',
            'comments': 'TOR, carbon that is measured after the introduction of helium/oxygen atmosphere at °550 C but beforetransmittance returns to initial value',
            'epa_code': '88336',
        },
        'OCf': {
            'name': 'carbon_organic_total_fine',
            'long_name': 'Carbon, Organic Total (Fine)',
            'comments': 'Organic carbon from TOR carbon fractions (OC1f+OC2f+OC3f+OC4f+OPf)',
            'epa_code': '88320',
        },
        'EC1f': {
            'name': 'carbon_elemental_fraction_1_fine',
            'long_name': 'Carbon, Elemental Fraction 1 (Fine)',
            'comments': 'TOR, 98% helium, 2% oxygen atmosphere, temperature (T) = 580° C.',
            'epa_code': '88329',
        },
        'EC2f': {
            'name': 'carbon_elemental_fraction_2_fine',
            'long_name': 'Carbon, Elemental Fraction 2 (Fine)',
            'comments': 'TOR, 98% helium, 2% oxygen atmosphere, temperature (T) = 740° C.',
            'epa_code': '88380',
        },
        'EC3f': {
            'name': 'carbon_elemental_fraction_3_fine',
            'long_name': 'Carbon, Elemental Fraction 3 (Fine)',
            'comments': 'TOR, 98% helium, 2% oxygen atmosphere, temperature (T) = 840° C.',
            'epa_code': '88331',
        },
        'ECf': {
            'name': 'carbon_elemental_total_fine',
            'long_name': 'Carbon, Elemental Total (Fine)',
            'comments': 'Elemental carbon from TOR carbon fractions (E1+E2+E3-OP)',
            'epa_code': '88321',
        },
        'fAbs': {
            'name': 'filter_absorption_coeff',
            'long_name': 'Filter Absorption Coefficient',
            'comments': 'A calibrated absorption coefficient measured from a Teflon filter using a hybrid integrating plate and sphere (HIPS) method',
            'epa_code': '63102',
        },
        'FlowRate': {
            'name': 'flow_rate',
            'long_name': 'Flow Rate',
            'comments': 'The rate of air flow through an air sampling instrument',
            'epa_code': '63102',
        },
        'MF': {
            'name': 'mass_pm2_5',
            'long_name': 'Mass, PM2.5 (Fine)',
            'comments': 'Gravimetric mass measurement for particles with aerodynamic diameters less than 2.5 um',
            'epa_code': '88101',
        },
        'MT': {
            'name': 'mass_pm10',
            'long_name': 'Mass, PM10 (Total)',
            'comments': 'Gravimetric mass measurement for particles with aerodynamic diameters less than 10 um',
            'epa_code': '85101',
        },
        'SampDur': {
            'name': 'sample_duration',
            'long_name': 'Sampling Duration',
            'comments': 'The duration of a given sampling period in minutes',
        },
        'ammNO3f': {
            'name': 'ammonium_nitrate_fine',
            'long_name': 'Ammonium Nitrate (Fine)',
            'comments': '1.29 x  NO3f',
        },
        'ammSO4f': {
            'name': 'ammonium_sulfate_fine',
            'long_name': 'Ammonium Sulfate (Fine)',
            'comments': '1.375 x SO4f',
        },
        'OMCf': {
            'name': 'carbon_organic_mass_fine',
            'long_name': 'Carbon, Organic Mass (fine)(1.8*OC)',
            'comments': '1.8 X OCf',
        },
        'TCf': {
            'name': 'carbon_total_fine',
            'long_name': 'Carbon, Total (fine)',
            'comments': 'From TOR carbon fractions (OCf+ECf)',
        },
        'CM_calculated': {
            'name': 'CM_calculated',
            'long_name': 'Mass, PM10-PM2.5 (Coarse)',
            'comments': 'MT-MF',
        },
        'SeaSaltf': {
            'name': 'sea_salt_fine',
            'long_name': 'Sea Salt (Fine)',
            'comments': '1.8XCHLf',
        },
        'SOILf': {
            'name': 'soil_fine',
            'long_name': 'Soil (Fine)',
            'comments': '2.2 × ALf + 2.49 × SIf + 1.63 × CAf + 2.42 × FEf + 1.94 × TIf',
        },
        'RCFM': {
            'name': 'mass_pm2_5_reconstructed',
            'long_name': 'Mass, PM2.5 Reconstructed (Fine)',
            'comments': 'Sum of ammSO4f, ammNO3f, OMCf, ECf, soilf, and seasaltf.',
        },
        'RCTM': {
            'name': 'mass_pmi10_reconstructed',
            'long_name': 'Mass, PM10 Reconstructed (Total)',
            'comments': 'Sum of ammSO4f, ammNO3f, OMCf, ECf, soilf, seasaltf, and CM_calculated.',
        },
    }
    laser_vars = {
        'RefF': {
            'units': 'ratio',
            'comments': 'Final laser reflectance at ',
            'name': 'final_laser_reflectance_',
        },
        'TransF': {
            'units': 'ratio',
            'comments': 'Final laser transmittance at ',
            'name': 'final_laser_transmittance_',
        },
        'RefI': {
            'units': 'ratio',
            'comments': 'Initial laser reflectance at ',
            'name': 'initial_laser_reflectance_',
        },
        'TransI': {
            'units': 'ratio',
            'comments': 'Initial laser transmittance at ',
            'name': 'initial_laser_transmittance_',
        },
        'RefM': {
            'units': 'ratio',
            'comments': 'Minimum laser reflectance at ',
            'name': 'min_laser_reflectance_',
        },
        'TransM': {
            'units': 'ratio',
            'comments': ' Minimum laser transmittance at ',
            'name': 'min_laser_transmittance_',
        },
        'OP_TR': {
            'units': 'ug m-3',
            'comments': 'Organic Pyrolyzed Carbon by Reflectance at ',
            'name': 'organic_pyrolyzed_carbon_reflectance_',
        },
        'OP_TT': {
            'units': 'ug m-3',
            'comments': 'Organic Pyrolyzed Carbon by Transmittance at ',
            'name': 'organic_pyrolyzed_carbon_transmittance_',
        },
    }
    laser_wl = ['405', '445', '532', '635', '780', '808', '980']
    for v in laser_vars:
        for wl in laser_wl:
            name = laser_vars[v]['name'] + wl
            if 'OP' not in v:
                mapping['_'.join([v, wl])] = {
                    'units': laser_vars[v]['units'],
                    'name': name,
                    'long_name': ' '.join([laser_vars[v]['comments'], wl]),
                }
            else:
                var_name = wl.join(v.split('_'))
                mapping[var_name] = {
                    'units': laser_vars[v]['units'],
                    'name': name,
                    'long_name': ' '.join([laser_vars[v]['comments'], wl]),
                }

    # Run through each variable in the dataframe and add it to a dataset
    # along with the appropriate metadata
    ct = 0
    site = np.unique(df.Site)[0]
    attrs = {'url': base_url, 'datastream': site + ' IMPROVE'}
    for v in variables:
        # Find data for just the variable in question
        poc_attrs = {'units': '1', 'long_name': 'Parameter Occurrence Code for ' + v}
        df2 = df[df.Param == v]

        # Get metadata
        unit = np.unique(df2.UnitAbbr)
        if len(unit) > 1:
            raise ValueError('Multiple types of units detected, using first one')

        sites = np.unique(df2.Site)
        if len(sites) > 1:
            raise ValueError('Multiple sites detected, please use only one')

        # Get time, POC, and data
        time = pd.to_datetime(df2.FactDate)
        poc = df2.POC

        data = df2.FactValue

        # Set up attributes
        var_attrs = {'units': unit[0], 'long_name': mapping[v]['long_name'], '_FillValue': -999.0}
        if 'comments' in mapping[v]:
            var_attrs['comments'] = mapping[v]['comments']
        if 'epa_code' in mapping[v]:
            var_attrs['epa_code'] = mapping[v]['epa_code']

        # If the first variable, create the dataset and then add variables to it
        if ct == 0:
            ds = xr.Dataset(
                data_vars={mapping[v]['name']: (['time'], data, var_attrs)},
                coords={'time': time},
                attrs=attrs,
            )
            ds['poc_' + mapping[v]['name']] = xr.DataArray(
                data=poc, dims=['time'], coords={'time': time}, attrs=poc_attrs
            )
            ct += 1
        else:
            ds[mapping[v]['name']] = xr.DataArray(
                data=data, dims=['time'], coords={'time': time}, attrs=var_attrs
            )
            ds['poc_' + mapping[v]['name']] = xr.DataArray(
                data=poc, dims=['time'], coords={'time': time}, attrs=poc_attrs
            )

    # Add in metadata from site summary page
    url = 'https://views.cira.colostate.edu/adms/Pub/SiteSummary.aspx?dsidse=10001&siidse=' + str(
        site_id
    )
    df = pd.read_html(url)
    for i in df[0].index:
        # Add lat/lon as variables
        if df[0][0][i] == 'Latitude':
            attrs = {
                'long_name': 'North latitude',
                'units': 'degree_N',
                'valid_min': -90.0,
                'valid_max': 90.0,
                'standard_name': 'latitude',
            }
            ds['lat'] = xr.DataArray(
                data=float(df[0][1][i]),
                dims=['time'],
                coords={'time': ds['time'].values},
                attrs=attrs,
            )
        elif df[0][0][i] == 'Longitude':
            attrs = {
                'long_name': 'East longitude',
                'units': 'degree_E',
                'valid_min': -180.0,
                'valid_max': 180.0,
                'standard_name': 'longitude',
            }
            ds['lon'] = xr.DataArray(
                data=float(df[0][1][i]),
                dims=['time'],
                coords={'time': ds['time'].values},
                attrs=attrs,
            )
        else:
            ds.attrs[df[0][0][i]] = df[0][1][i]

    # Add in problem information from the site summary page
    problem = ''
    for i in df[-1].index:
        problem += '_'.join(
            [df[-1]['EventDate'][i], df[-1]['EventType'][i], df[-1]['Notes'][i], '\n']
        )
    ds.attrs['site_problems'] = problem

    return ds
