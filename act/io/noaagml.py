"""
Modules for reading in NOAA GML data

"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .text import read_csv


def read_gml(filename, datatype=None, remove_time_vars=True, convert_missing=True, **kwargs):
    """
    Function to call or guess what reading NOAA GML daga routine to use. It
    tries to guess the correct reading function to call based on filename.
    It mostly works, but you may want to specify for best results.

    Parameters
    ----------
    filename : str or pathlib.Path
        Data file full path name. In theory it should work with a list of
        filenames but it is not working as well with that as expected.
    datatype : str
        Data file type that bypasses the guessing from filename format
        and goes directly to the reading routine. Options include
        [MET, RADIATION, OZONE, CO2, HALO]
    remove_time_vars : bool
        Some variables are convereted into coordinate variables in Xarray
        DataSet and not needed after conversion. This will remove those
        variables.
    convert_missing : bool
        Convert missing value indicator in CSV to NaN in Xarray DataSet.
    **kwargs : keywords
        Keywords to pass through to instrument specific reading routine.

    Returns
    -------
    ds : xarray.Dataset
        Standard ARM Xarray dataset with the data cleaned up to have units,
        long_name, correct type and some other stuff.

    """

    if datatype is not None:
        if datatype.upper() == 'MET':
            return read_gml_met(filename, convert_missing=convert_missing, **kwargs)
        elif datatype.upper() == 'RADIATION':
            return read_gml_radiation(
                filename,
                remove_time_vars=remove_time_vars,
                convert_missing=convert_missing,
                **kwargs,
            )
        elif datatype.upper() == 'OZONE':
            return read_gml_ozone(filename, **kwargs)
        elif datatype.upper() == 'CO2':
            return read_gml_co2(filename, convert_missing=convert_missing, **kwargs)
        elif datatype.upper() == 'HALO':
            return read_gml_halo(filename, **kwargs)
        else:
            raise ValueError('datatype is unknown')

    else:
        test_filename = filename
        if isinstance(test_filename, (list, tuple)):
            test_filename = filename[0]

        test_filename = str(Path(test_filename).name)

        if test_filename.startswith('met_') and test_filename.endswith('.txt'):
            return read_gml_met(filename, convert_missing=convert_missing, **kwargs)

        if test_filename.startswith('co2_') and test_filename.endswith('.txt'):
            return read_gml_co2(filename, convert_missing=convert_missing, **kwargs)

        result = re.match(r'([a-z]{3})([\d]{5}).dat', test_filename)
        if result is not None:
            return read_gml_radiation(
                filename,
                remove_time_vars=remove_time_vars,
                convert_missing=convert_missing,
                **kwargs,
            )

        ozone_pattern = [
            r'[a-z]{3}_[\d]{4}_[\d]{2}_hour.dat',
            r'[a-z]{3}_[\d]{2}_[\d]{4}_hour.dat',
            r'[a-z]{3}_[\d]{4}_all_minute.dat',
            r'[a-z]{3}_[\d]{2}_[\d]{4}_5minute.dat',
            r'[a-z]{3}_[\d]{2}_[\d]{4}_min.dat',
            r'[a-z]{3}_o3_6m_hour_[\d]{2}_[\d]{4}.dat',
            r'[a-z]{3}_ozone_houry__[\d]{4}',
        ]
        for pattern in ozone_pattern:
            result = re.match(pattern, test_filename)
            if result is not None:
                return read_gml_ozone(filename, **kwargs)

        ozone_pattern = [
            r'[a-z]{3}_CCl4_Day.dat',
            r'[a-z]{3}_CCl4_All.dat',
            r'[a-z]{3}_CCl4_MM.dat',
            r'[a-z]{3}_MC_MM.dat',
        ]
        for pattern in ozone_pattern:
            result = re.match(pattern, test_filename)
            if result is not None:
                return read_gml_halo(filename, **kwargs)


def read_gml_halo(filename, **kwargs):
    """
    Function to read Halocarbon data from NOAA GML.

    Parameters
    ----------
    filename : str or pathlib.Path
        Data file full path name.

    Returns
    -------
    ds : xarray.Dataset
        Standard ARM Xarray dataset with the data cleaned up to have units,
        long_name, correct type and some other stuff.
    **kwargs : keywords
        Keywords to pass through to ACT read_csv() routine.

    """
    ds = None
    if filename is None:
        return ds

    variables = {
        'CCl4catsBRWm': {
            'long_name': 'Carbon Tetrachloride (CCl4) daily median',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'CCl4',
        },
        'CCl4catsBRWmsd': {
            'long_name': 'Carbon Tetrachloride (CCl4) standard deviation',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'CCl4_std_dev',
        },
        'CCl4catsBRWsd': {
            'long_name': 'Carbon Tetrachloride (CCl4) standard deviation',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'CCl4_std_dev',
        },
        'CCl4catsBRWn': {
            'long_name': 'Number of samples',
            'units': 'count',
            '__type': np.int16,
            '__rename': 'number_of_samples',
        },
        'CCl4catsBRWunc': {
            'long_name': 'Carbon Tetrachloride (CCl4) uncertainty',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'CCl4_uncertainty',
        },
        'MCcatsBRWm': {
            'long_name': 'Methyl Chloroform (CH3CCl3)',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'methyl_chloroform',
        },
        'MCcatsBRWunc': {
            'long_name': 'Methyl Chloroform (CH3CCl3) uncertainty',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'methyl_chloroform_uncertainty',
        },
        'MCcatsBRWsd': {
            'long_name': 'Methyl Chloroform (CH3CCl3) standard deviation',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'methyl_chloroform_std_dev',
        },
        'MCcatsBRWmsd': {
            'long_name': 'Methyl Chloroform (CH3CCl3) standard deviation',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'methyl_chloroform_std_dev',
        },
        'MCcatsBRWn': {
            'long_name': 'Number of samples',
            'units': 'count',
            '__type': np.int16,
            '__rename': 'number_of_samples',
        },
        'MCritsBRWm': {
            'long_name': 'Methyl Chloroform (CH3CCl3)',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'methyl_chloroform',
        },
        'MCritsBRWsd': {
            'long_name': 'Methyl Chloroform (CH3CCl3) standard deviation',
            'units': 'ppt',
            '_FillValue': np.nan,
            '__type': np.float32,
            '__rename': 'methyl_chloroform_std_dev',
        },
        'MCritsBRWn': {
            'long_name': 'Number of samples',
            'units': 'count',
            '__type': np.int16,
            '__rename': 'number_of_samples',
        },
    }

    test_filename = filename
    if isinstance(test_filename, (list, tuple)):
        test_filename = test_filename[0]

    with open(test_filename) as fc:
        header = 0
        while True:
            line = fc.readline().strip()
            if not line.startswith('#'):
                break
            header += 1

    ds = read_csv(
        filename, sep=r'\s+', header=header, na_values=['Nan', 'NaN', 'nan', 'NAN'], **kwargs
    )
    var_names = list(ds.data_vars)
    year_name, month_name, day_name, hour_name, min_name = None, None, None, None, None
    for var_name in var_names:
        if var_name.endswith('yr'):
            year_name = var_name
        elif var_name.endswith('mon'):
            month_name = var_name
        elif var_name.endswith('day'):
            day_name = var_name
        elif var_name.endswith('hour'):
            hour_name = var_name
        elif var_name.endswith('min'):
            min_name = var_name

    timestamp = np.full(ds[var_names[0]].size, np.nan, dtype='datetime64[ns]')
    for ii in range(0, len(timestamp)):
        if min_name is not None:
            ts = datetime(
                ds[year_name].values[ii],
                ds[month_name].values[ii],
                ds[day_name].values[ii],
                ds[hour_name].values[ii],
                ds[min_name].values[ii],
            )
        elif hour_name is not None:
            ts = datetime(
                ds[year_name].values[ii],
                ds[month_name].values[ii],
                ds[day_name].values[ii],
                ds[hour_name].values[ii],
            )
        elif day_name is not None:
            ts = datetime(
                ds[year_name].values[ii],
                ds[month_name].values[ii],
                ds[day_name].values[ii],
            )
        else:
            ts = datetime(ds[year_name].values[ii], ds[month_name].values[ii], 1)

        timestamp[ii] = np.datetime64(ts, 'ns')

    for var_name in [year_name, month_name, day_name, hour_name, min_name]:
        try:
            del ds[var_name]
        except KeyError:
            pass

    ds = ds.rename({'index': 'time'})
    ds = ds.assign_coords(time=timestamp)
    ds['time'].attrs['long_name'] = 'Time'

    for var_name, value in variables.items():
        if var_name not in var_names:
            continue

        for att_name, att_value in value.items():
            if att_name == '__type':
                values = ds[var_name].values
                values = values.astype(att_value)
                ds[var_name].values = values
            elif att_name == '__rename':
                ds = ds.rename({var_name: att_value})
            else:
                ds[var_name].attrs[att_name] = att_value

    return ds


def read_gml_co2(filename=None, convert_missing=True, **kwargs):
    """
    Function to read carbon dioxide data from NOAA GML.

    Parameters
    ----------
    filename : str or pathlib.Path
        Data file full path name.
    convert_missing : boolean
        Option to convert missing values to NaN. If turned off will
        set variable attribute to missing value expected. This works well
        to preserve the data type best for writing to a netCDF file.

    Returns
    -------
    ds : xarray.Dataset
        Standard ARM Xarray dataset with the data cleaned up to have units,
        long_name, correct type and some other stuff.
    **kwargs : keywords
        Keywords to pass through to ACT read_csv() routine.

    """
    ds = None
    if filename is None:
        return ds

    variables = {
        'site_code': None,
        'year': None,
        'month': None,
        'day': None,
        'hour': None,
        'minute': None,
        'second': None,
        'time_decimal': None,
        'value': {
            'long_name': 'Carbon monoxide in dry air',
            'units': 'ppm',
            '_FillValue': -999.99,
            'comment': (
                'Mole fraction reported in units of micromol mol-1 '
                '(10-6 mol per mol of dry air); abbreviated as ppm (parts per million).'
            ),
            '__type': np.float32,
            '__rename': 'co2',
        },
        'value_std_dev': {
            'long_name': 'Carbon monoxide in dry air',
            'units': 'ppm',
            '_FillValue': -99.99,
            'comment': (
                'This is the standard deviation of the reported mean value '
                'when nvalue is greater than 1. See provider_comment if available.'
            ),
            '__type': np.float32,
            '__rename': 'co2_std_dev',
        },
        'nvalue': {
            'long_name': 'Number of measurements contributing to reported value',
            'units': '1',
            '_FillValue': -9,
            '__type': np.int16,
            '__rename': 'number_of_measurements',
        },
        'latitude': {
            'long_name': 'Latitude at which air sample was collected',
            'units': 'degrees_north',
            '_FillValue': -999.999,
            'standard_name': 'latitude',
            '__type': np.float32,
        },
        'longitude': {
            'long_name': 'Latitude at which air sample was collected',
            'units': 'degrees_east',
            '_FillValue': -999.999,
            'standard_name': 'longitude',
            '__type': np.float32,
        },
        'altitude': {
            'long_name': 'Sample altitude',
            'units': 'm',
            '_FillValue': -999.999,
            'standard_name': 'altitude',
            'comment': (
                'Altitude for this dataset is the sum of surface elevation '
                '(masl) and sample intake height (magl)'
            ),
            '__type': np.float32,
        },
        'intake_height': {
            'long_name': 'Sample intake height above ground level',
            'units': 'm',
            '_FillValue': -999.999,
            '__type': np.float32,
        },
    }

    test_filename = filename
    if isinstance(test_filename, (list, tuple)):
        test_filename = test_filename[0]

    with open(test_filename) as fc:
        skiprows = int(fc.readline().strip().split()[-1]) - 1

    ds = read_csv(filename, sep=r'\s+', skiprows=skiprows, **kwargs)

    timestamp = np.full(ds['year'].size, np.nan, dtype='datetime64[ns]')
    for ii in range(0, len(timestamp)):
        ts = datetime(
            ds['year'].values[ii],
            ds['month'].values[ii],
            ds['day'].values[ii],
            ds['hour'].values[ii],
            ds['minute'].values[ii],
            ds['second'].values[ii],
        )
        timestamp[ii] = np.datetime64(ts, 'ns')

    ds = ds.rename({'index': 'time'})
    ds = ds.assign_coords(time=timestamp)
    ds['time'].attrs['long_name'] = 'Time'

    for var_name, value in variables.items():
        if value is None:
            del ds[var_name]
        else:
            for att_name, att_value in value.items():
                if att_name == '__type':
                    values = ds[var_name].values
                    values = values.astype(att_value)
                    ds[var_name].values = values
                elif att_name == '__rename':
                    ds = ds.rename({var_name: att_value})
                else:
                    ds[var_name].attrs[att_name] = att_value

            if convert_missing:
                try:
                    var_name = variables[var_name]['__rename']
                except KeyError:
                    pass

                try:
                    missing_value = ds[var_name].attrs['_FillValue']
                    values = ds[var_name].values.astype(float)
                    values[np.isclose(missing_value, values)] = np.nan
                    ds[var_name].values = values
                    ds[var_name].attrs['_FillValue'] = np.nan
                except KeyError:
                    pass

    values = ds['qcflag'].values
    bad_index = []
    suspect_index = []
    for ii, value in enumerate(values):
        pts = list(value)
        if pts[0] != '.':
            bad_index.append(ii)
        if pts[1] != '.':
            suspect_index.append(ii)

    var_name = 'co2'
    qc_var_name = ds.qcfilter.create_qc_variable(var_name)
    ds.qcfilter.add_test(
        var_name,
        index=bad_index,
        test_assessment='Bad',
        test_meaning='Obvious problems during collection or analysis',
    )
    ds.qcfilter.add_test(
        var_name,
        index=suspect_index,
        test_assessment='Indeterminate',
        test_meaning=(
            'Likely valid but does not meet selection criteria determined by '
            'the goals of a particular investigation'
        ),
    )
    ds[qc_var_name].attrs[
        'comment'
    ] = 'This quality control flag is provided by the contributing PIs'
    del ds['qcflag']

    return ds


def read_gml_ozone(filename=None, **kwargs):
    """
    Function to read ozone data from NOAA GML.

    Parameters
    ----------
    filename : str or pathlib.Path
        Data file full path name.
    **kwargs : keywords
        Keywords to pass through to ACT read_csv() routine.

    Returns
    -------
    ds : xarray.Dataset
        Standard ARM Xarray dataset with the data cleaned up to have units,
        long_name, correct type and some other stuff.

    """
    ds = None
    if filename is None:
        return ds

    test_filename = filename
    if isinstance(test_filename, (list, tuple)):
        test_filename = test_filename[0]

    with open(test_filename) as fc:
        skiprows = 0
        while True:
            line = fc.readline().strip().split()
            try:
                if len(line) == 6 and line[0] == 'STN':
                    break
            except IndexError:
                pass
            skiprows += 1

    ds = read_csv(filename, sep=r'\s+', skiprows=skiprows, **kwargs)
    ds.attrs['station'] = str(ds['STN'].values[0]).lower()

    timestamp = np.full(ds['YEAR'].size, np.nan, dtype='datetime64[ns]')
    for ii in range(0, len(timestamp)):
        ts = datetime(
            ds['YEAR'].values[ii],
            ds['MON'].values[ii],
            ds['DAY'].values[ii],
            ds['HR'].values[ii],
        )
        timestamp[ii] = np.datetime64(ts, 'ns')

    ds = ds.rename({'index': 'time'})
    ds = ds.assign_coords(time=timestamp)
    ds['time'].attrs['long_name'] = 'Time'

    for var_name in ['STN', 'YEAR', 'MON', 'DAY', 'HR']:
        del ds[var_name]

    var_name = 'ozone'
    ds = ds.rename({'O3(PPB)': var_name})
    ds[var_name].attrs['long_name'] = 'Ozone'
    ds[var_name].attrs['units'] = 'ppb'
    ds[var_name].attrs['_FillValue'] = np.nan
    ds[var_name].values = ds[var_name].values.astype(np.float32)

    return ds


def read_gml_radiation(filename=None, convert_missing=True, remove_time_vars=True, **kwargs):
    """
    Function to read radiation data from NOAA GML.

    Parameters
    ----------
    filename : str or pathlib.Path
        Data file full path name.
    convert_missing : boolean
        Option to convert missing values to NaN. If turned off will
        set variable attribute to missing value expected. This works well
        to preserve the data type best for writing to a netCDF file.
    remove_time_vars : boolean
        Some column names in the CSV file are used for creating the time
        coordinate variable in the returend Xarray DataSet. Once used the
        variables are not needed and will be removed from DataSet.
    **kwargs : keywords
        Keywords to pass through to ACT read_csv() routine.

    Returns
    -------
    ds : xarray.Dataset
        Standard ARM Xarray dataset with the data cleaned up to have units,
        long_name, correct type and some other stuff.
    """

    ds = None
    if filename is None:
        return ds

    column_names = {
        'year': None,
        'jday': None,
        'month': None,
        'day': None,
        'hour': None,
        'minute': None,
        'decimal_time': None,
        'solar_zenith_angle': {
            'units': 'degree',
            'long_name': 'Solar zenith angle',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'downwelling_global_solar': {
            'units': 'W/m^2',
            'long_name': 'Downwelling global solar',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'upwelling_global_solar': {
            'units': 'W/m^2',
            'long_name': 'Upwelling global solar',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'direct_normal_solar': {
            'units': 'W/m^2',
            'long_name': 'Direct-normal solar',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'downwelling_diffuse_solar': {
            'units': 'W/m^2',
            'long_name': 'Downwelling diffuse solar',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'downwelling_thermal_infrared': {
            'units': 'W/m^2',
            'long_name': 'Downwelling thermal infrared',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'downwelling_infrared_case_temp': {
            'units': 'degK',
            'long_name': 'Downwelling infrared case temp',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'downwelling_infrared_dome_temp': {
            'units': 'degK',
            'long_name': 'downwelling infrared dome temp',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'upwelling_thermal_infrared': {
            'units': 'W/m^2',
            'long_name': 'Upwelling thermal infrared',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'upwelling_infrared_case_temp': {
            'units': 'degK',
            'long_name': 'Upwelling infrared case temp',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'upwelling_infrared_dome_temp': {
            'units': 'degK',
            'long_name': 'Upwelling infrared dome temp',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'global_UVB': {
            'units': 'mW/m^2',
            'long_name': 'global ultraviolet-B',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'par': {
            'units': 'W/m^2',
            'long_name': 'Photosynthetically active radiation',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'net_solar': {
            'units': 'W/m^2',
            'long_name': 'Net solar (downwelling_global_solar - upwelling_global_solar)',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'net_infrared': {
            'units': 'W/m^2',
            'long_name': (
                'Net infrared (downwelling_thermal_infrared - ' 'upwelling_thermal_infrared)'
            ),
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'net_radiation': {
            'units': 'W/m^2',
            'long_name': 'Net radiation (net_solar + net_infrared)',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'air_temperature_10m': {
            'units': 'degC',
            'long_name': '10-meter air temperature',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'relative_humidity': {
            'units': '%',
            'long_name': 'Relative humidity',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'wind_speed': {
            'units': 'm/s',
            'long_name': 'Wind speed',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'wind_direction': {
            'units': 'degree',
            'long_name': 'Wind direction (clockwise from north)',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
        'station_pressure': {
            'units': 'millibar',
            'long_name': 'Station atmospheric pressure',
            '_FillValue': -9999.9,
            '__type': np.float32,
        },
    }

    # Add additinal column names for NOAA SPASH campaign
    if str(Path(filename).name).startswith('cbc') or str(Path(filename).name).startswith('ckp'):
        column_names['SPN1_total'] = {
            'units': 'W/m^2',
            'long_name': 'SPN1 total average',
            '_FillValue': -9999.9,
            '__type': np.float32,
        }
        column_names['SPN1_diffuse'] = {
            'units': 'W/m^2',
            'long_name': 'SPN1 diffuse average',
            '_FillValue': -9999.9,
            '__type': np.float32,
        }

    names = list(column_names.keys())
    skip_vars = [
        'year',
        'jday',
        'month',
        'day',
        'hour',
        'minute',
        'decimal_time',
        'solar_zenith_angle',
    ]
    num = 1
    for ii, name in enumerate(column_names.keys()):
        if name in skip_vars:
            continue
        names.insert(ii + num, 'qc_' + name)
        num += 1

    ds = read_csv(filename, sep=r'\s+', header=None, skiprows=2, column_names=names, **kwargs)

    if isinstance(filename, (list, tuple)):
        filename = filename[0]

    if ds is not None:
        with open(filename) as fc:
            lat = None
            lon = None
            alt = None
            alt_unit = None
            station = None
            for ii in [0, 1]:
                line = fc.readline().strip().split()
                if len(line) == 1:
                    station = line[0]
                else:
                    lat = np.array(line[0], dtype=np.float32)
                    lon = np.array(line[1], dtype=np.float32)
                    alt = np.array(line[2], dtype=np.float32)
                    alt_unit = str(line[3])

        ds['lat'] = xr.DataArray(
            lat,
            attrs={
                'long_name': 'Latitude',
                'units': 'degree_north',
                'standard_name': 'latitude',
            },
        )
        ds['lon'] = xr.DataArray(
            lon,
            attrs={
                'long_name': 'Longitude',
                'units': 'degree_east',
                'standard_name': 'longitude',
            },
        )
        ds['alt'] = xr.DataArray(
            alt,
            attrs={
                'long_name': 'Latitude',
                'units': alt_unit,
                'standard_name': 'altitude',
            },
        )
        ds.attrs['location'] = station

        timestamp = np.full(ds['year'].size, np.nan, dtype='datetime64[ns]')
        for ii in range(0, len(timestamp)):
            ts = datetime(
                ds['year'].values[ii],
                ds['month'].values[ii],
                ds['day'].values[ii],
                ds['hour'].values[ii],
                ds['minute'].values[ii],
            )
            timestamp[ii] = np.datetime64(ts, 'ns')

        ds = ds.rename({'index': 'time'})
        ds = ds.assign_coords(time=timestamp)
        ds['time'].attrs['long_name'] = 'Time'
        for var_name, value in column_names.items():
            if value is None:
                ds[var_name]
            else:
                for att_name, att_value in value.items():
                    if att_name == '__type':
                        values = ds[var_name].values
                        values = values.astype(att_value)
                        ds[var_name].values = values
                    else:
                        ds[var_name].attrs[att_name] = att_value

                if convert_missing:
                    try:
                        missing_value = ds[var_name].attrs['_FillValue']
                        values = ds[var_name].values.astype(float)
                        index = np.isclose(values, missing_value)
                        values[index] = np.nan
                        ds[var_name].values = values
                        ds[var_name].attrs['_FillValue'] = np.nan
                    except KeyError:
                        pass

        for var_name in ds.data_vars:
            if not var_name.startswith('qc_'):
                continue
            data_var_name = var_name.replace('qc_', '', 1)
            attrs = {
                'long_name': f"Quality control variable for: {ds[data_var_name].attrs['long_name']}",
                'units': '1',
                'standard_name': 'quality_flag',
                'flag_values': [0, 1, 2],
                'flag_meanings': [
                    'Not failing any tests',
                    'Knowingly bad value',
                    'Should be used with scrutiny',
                ],
                'flag_assessments': ['Good', 'Bad', 'Indeterminate'],
            }
            ds[var_name].attrs = attrs
            ds[data_var_name].attrs['ancillary_variables'] = var_name

        if remove_time_vars:
            remove_var_names = [
                'year',
                'jday',
                'month',
                'day',
                'hour',
                'minute',
                'decimal_time',
            ]
            ds = ds.drop_vars(remove_var_names)

    return ds


def read_gml_met(filename=None, convert_missing=True, **kwargs):
    """
    Function to read meteorological data from NOAA GML.

    Parameters
    ----------
    filename : str or pathlib.Path
        Data file full path name.
    convert_missing : boolean
        Option to convert missing values to NaN. If turned off will
        set variable attribute to missing value expected. This works well
        to preserve the data type best for writing to a netCDF file.
    **kwargs : keywords
        Keywords to pass through to ACT read_csv() routine.

    Returns
    -------
    ds : xarray.Dataset
        Standard ARM Xarray dataset with the data cleaned up to have units,
        long_name, correct type and some other stuff.
    """

    ds = None
    if filename is None:
        return ds

    column_names = {
        'station': None,
        'year': None,
        'month': None,
        'day': None,
        'hour': None,
        'minute': None,
        'wind_direction': {
            'units': 'degree',
            'long_name': 'Average wind direction from which the wind is blowing',
            '_FillValue': -999,
            '__type': np.int16,
        },
        'wind_speed': {
            'units': 'm/s',
            'long_name': 'Average wind speed',
            '_FillValue': -999.9,
            '__type': np.float32,
        },
        'wind_steadiness_factor': {
            'units': '1',
            'long_name': '100 times the ratio of the vector wind speed to the '
            'average wind speed for the hour',
            '_FillValue': -9,
            '__type': np.int16,
        },
        'barometric_pressure': {
            'units': 'hPa',
            'long_name': 'Station barometric pressure',
            '_FillValue': -999.90,
            '__type': np.float32,
        },
        'temperature_2m': {
            'units': 'degC',
            'long_name': 'Temperature at 2 meters above ground level',
            '_FillValue': -999.9,
            '__type': np.float32,
        },
        'temperature_10m': {
            'units': 'degC',
            'long_name': 'Temperature at 10 meters above ground level',
            '_FillValue': -999.9,
            '__type': np.float32,
        },
        'temperature_tower_top': {
            'units': 'degC',
            'long_name': 'Temperature at top of instrument tower',
            '_FillValue': -999.9,
            '__type': np.float32,
        },
        'realitive_humidity': {
            'units': 'percent',
            'long_name': 'Relative humidity',
            '_FillValue': -99,
            '__type': np.int16,
        },
        'preciptation_intensity': {
            'units': 'mm/hour',
            'long_name': 'Amount of precipitation per hour',
            '_FillValue': -99,
            '__type': np.int16,
            'comment': (
                'The precipitation amount is measured with an unheated '
                'tipping bucket rain gauge.'
            ),
        },
    }

    minutes = True
    test_filename = filename
    if isinstance(test_filename, (list, tuple)):
        test_filename = test_filename[0]

    if '_hour_' in Path(test_filename).name:
        minutes = False
        del column_names['minute']

    ds = read_csv(filename, sep=r'\s+', header=None, column_names=column_names.keys(), **kwargs)

    if ds is not None:
        timestamp = np.full(ds['year'].size, np.nan, dtype='datetime64[ns]')
        for ii in range(0, len(timestamp)):
            if minutes:
                ts = datetime(
                    ds['year'].values[ii],
                    ds['month'].values[ii],
                    ds['day'].values[ii],
                    ds['hour'].values[ii],
                    ds['minute'].values[ii],
                )
            else:
                ts = datetime(
                    ds['year'].values[ii],
                    ds['month'].values[ii],
                    ds['day'].values[ii],
                    ds['hour'].values[ii],
                )

            timestamp[ii] = np.datetime64(ts, 'ns')

        ds = ds.rename({'index': 'time'})
        ds = ds.assign_coords(time=timestamp)
        ds['time'].attrs['long_name'] = 'Time'
        for var_name, value in column_names.items():
            if value is None:
                del ds[var_name]
            else:
                for att_name, att_value in value.items():
                    if att_name == '__type':
                        values = ds[var_name].values
                        values = values.astype(att_value)
                        ds[var_name].values = values
                    else:
                        ds[var_name].attrs[att_name] = att_value

                if convert_missing:
                    try:
                        missing_value = ds[var_name].attrs['_FillValue']
                        values = ds[var_name].values.astype(float)
                        index = np.isclose(values, missing_value)
                        values[index] = np.nan
                        ds[var_name].values = values
                        ds[var_name].attrs['_FillValue'] = np.nan
                    except KeyError:
                        pass

    return ds


def read_surfrad(filename, **kwargs):
    """
    Function to read in NOAA SurfRad data

    Parameters
    ----------
    filename : list
        Data files full path name or url to file
    **kwargs : keywords
        Keywords to pass through to instrument specific reading routine.

    Returns
    -------
    ds : xarray.Dataset
        Standard ARM Xarray dataset with the data cleaned up to have units,
        long_name, correct type and some other stuff.

    """

    names = [
        'year',
        'jday',
        'month',
        'day',
        'hour',
        'minute',
        'dec_time',
        'solar_zenith_angle',
        'downwelling_global',
        'qc_downwelling_global',
        'upwelling_global',
        'qc_upwelling_global',
        'direct_normal',
        'qc_direct_normal',
        'downwelling_diffuse',
        'qc_downwelling_diffuse',
        'downwelling_ir',
        'qc_downwelling_ir',
        'downwelling_ir_casetemp',
        'qc_downwelling_ir_casetemp',
        'downwelling_ir_dometemp',
        'qc_downwelling_ir_dometemp',
        'upwelling_ir',
        'qc_upwelling_ir',
        'upwelling_ir_casetemp',
        'qc_upwelling_ir_casetemp',
        'upwelling_ir_dometemp',
        'qc_upwelling_ir_dometemp',
        'global_uvb',
        'qc_global_uvb',
        'par',
        'qc_par',
        'net_radiation',
        'qc_net_radiation',
        'net_ir',
        'qc_net_ir',
        'total_net',
        'qc_total_net',
        'temperature',
        'qc_temperature',
        'relative_humidity',
        'qc_relative_humidity',
        'wind_speed',
        'qc_wind_speed',
        'wind_direction',
        'qc_wind_direction',
        'pressure',
        'qc_pressure',
    ]
    for i, f in enumerate(filename):
        new_df = pd.read_csv(f, names=names, skiprows=2, delimiter=r'\s+', header=None)
        if i == 0:
            df = new_df
        else:
            df = pd.concat([df, new_df])

    # Create time variable and add as the coordinate
    ds = df.to_xarray()
    year = ds['year'].values
    month = ds['month'].values
    day = ds['day'].values
    hour = ds['hour'].values
    minute = ds['minute'].values
    time = [datetime(year[i], month[i], day[i], hour[i], minute[i]) for i in range(len(year))]
    ds = ds.assign_coords(index=time)
    ds = ds.rename(index='time')

    # Add attributes
    attrs = {
        'year': {'long_name': 'Year', 'units': 'unitless'},
        'jday': {'long_name': 'Julian day', 'units': 'unitless'},
        'month': {'long_name': 'Month', 'units': 'unitless'},
        'day': {'long_name': 'Day of the month', 'units': 'unitless'},
        'hour': {'long_name': 'Hour', 'units': 'unitless'},
        'minute': {'long_name': 'Minutes', 'units': 'unitless'},
        'dec_time': {'long_name': 'Decimal time', 'units': 'unitless'},
        'solar_zenith_angle': {'long_name': 'Solar zenith angle', 'units': 'deg'},
        'downwelling_global': {
            'long_name': 'Downwelling global solar',
            'units': 'W m^-2',
            'standard_name': 'surface_downwelling_shortwave_flux_in_air',
        },
        'upwelling_global': {
            'long_name': 'Upwelling global solar',
            'units': 'W m^-2',
            'standard_name': 'surface_upwelling_shortwave_flux_in_air',
        },
        'direct_normal': {
            'long_name': 'Direct normal solar',
            'units': 'W m^-2',
            'standard_name': 'surface_direct_downwelling_shortwave_flux_in_air',
        },
        'downwelling_diffuse': {
            'long_name': 'Downwelling diffuse solar',
            'units': 'W m^-2',
            'standard_name': 'diffuse_downwelling_shortwave_flux_in_air',
        },
        'downwelling_ir': {
            'long_name': 'Downwelling thermal infrared',
            'units': 'W m^-2',
            'standard_name': 'net_downward_longwave_flux_in_air',
        },
        'downwelling_ir_casetemp': {
            'long_name': 'Downwelling thermal infrared case temperature',
            'units': 'K',
        },
        'downwelling_ir_dometemp': {
            'long_name': 'Downwelling thermal infrared dome temperature',
            'units': 'K',
        },
        'upwelling_ir': {
            'long_name': 'Upwelling thermal infrared',
            'units': 'W m^-2',
            'standard_name': 'net_upward_longwave_flux_in_air',
        },
        'upwelling_ir_casetemp': {
            'long_name': 'Upwelling thermal infrared case temperature',
            'units': 'K',
        },
        'upwelling_ir_dometemp': {
            'long_name': 'Upwelling thermal infrared dome temperature',
            'units': 'K',
        },
        'global_uvb': {'long_name': 'Global UVB', 'units': 'milliWatts m^-2'},
        'par': {
            'long_name': 'Photosynthetically active radiation',
            'units': 'W m^-2',
            'standard_name': 'surface_downwelling_photosynthetic_radiative_flux_in_air',
        },
        'net_radiation': {
            'long_name': 'Net solar (downwelling_global-upwelling_global)',
            'units': 'W m^-2',
            'standard_name': 'surface_net_downward_shortwave_flux',
        },
        'net_ir': {
            'long_name': 'Net infrared (downwelling_ir-upwelling_ir)',
            'units': 'W m^-2',
            'standard_name': 'surface_net_downward_longwave_flux',
        },
        'total_net': {
            'long_name': 'Total Net radiation (net_radiation + net_ir)',
            'units': 'W m^-2',
        },
        'temperature': {
            'long_name': '10-meter air temperature',
            'units': 'degC',
            'standard_name': 'air_temperature',
        },
        'relative_humidity': {
            'long_name': 'Relative humidity',
            'units': '%',
            'standard_name': 'relative_humidity',
        },
        'wind_speed': {'long_name': 'Wind speed', 'units': 'ms^-1', 'standard_name': 'wind_speed'},
        'wind_direction': {
            'long_name': 'Wind direction, clockwise from North',
            'units': 'deg',
            'standard_name': 'wind_from_direction',
        },
        'pressure': {
            'long_name': 'Station pressure',
            'units': 'mb',
            'standard_name': 'air_pressure',
        },
    }

    for v in ds:
        if v in attrs:
            ds[v].attrs = attrs[v]

    # Add attributes to all QC variables
    qc_vars = [
        'downwelling_global',
        'upwelling_global',
        'direct_normal',
        'downwelling_diffuse',
        'downwelling_ir',
        'downwelling_ir_casetemp',
        'downwelling_ir_dometemp',
        'upwelling_ir',
        'upwelling_ir_casetemp',
        'upwelling_ir_dometemp',
        'global_uvb',
        'par',
        'net_radiation',
        'net_ir',
        'total_net',
        'temperature',
        'relative_humidity',
        'wind_speed',
        'wind_direction',
        'pressure',
    ]

    for v in qc_vars:
        atts = {
            'long_name': 'Quality check results on variable: ' + v,
            'units': '1',
            'description': ''.join(
                [
                    'A QC flag of zero indicates that the corresponding data point is good,',
                    ' having passed all QC checks.  A value greater than 0 indicates that',
                    ' the data failed one level of QC.  For example, a QC value of 1 means',
                    ' that the recorded value is beyond a physically possible range, or it has',
                    ' been affected adversely in some manner to produce a knowingly bad value.',
                    ' A value of 2 indicates that the data value failed the second level QC check,',
                    ' indicating that the data value may be physically possible but should be used',
                    ' with scrutiny, and so on.',
                ]
            ),
        }
        ds['qc_' + v].attrs = atts

    ds.attrs['datastream'] = 'SURFRAD Site: ' + filename[0].split('/')[-1][0:3]

    return ds
