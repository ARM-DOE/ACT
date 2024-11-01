"""
Consolidation of Data Sources
-----------------------------

This example shows how to use ACT to combine multiple
datasets to support ARM's AMF3.

"""

import act
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

# Get Surface Meteorology data
station = '1M4'
time_window = [datetime(2024, 10, 24), datetime(2024, 10, 31)]
ds_asos = act.discovery.get_asos_data(time_window, station=station, regions='AL')[station]
ds_asos = ds_asos.where(~np.isnan(ds_asos.tmpf), drop=True)
ds_asos['tmpf'].attrs['units'] = 'degF'
ds_asos.utils.change_units(variables='tmpf', desired_unit='degC', verbose=True)

# You need an account and token from https://docs.airnowapi.org/ first
airnow_token = os.getenv('AIRNOW_API')
if airnow_token is not None and len(airnow_token) > 0:
    latlon = '-87.453,34.179,-86.477,34.787'
    ds_airnow = act.discovery.get_airnow_bounded_obs(
        airnow_token, '2024-10-24T00', '2024-10-31T23', latlon, 'OZONE,PM25', data_type='B'
    )
    ds_airnow = act.utils.convert_2d_to_1d(ds_airnow, parse='sites')
    sites = ds_airnow['sites'].values

# Get NOAA Data
results = act.discovery.download_noaa_psl_data(
    site='ctd', instrument='Temp/RH', startdate='20241024', enddate='20241031'
)
ds_noaa = act.io.read_psl_surface_met(results)

# Place your username and token here
username = os.getenv('ARM_USERNAME')
token = os.getenv('ARM_PASSWORD')

# If the username and token are not set, use the existing sample file
if username is not None and token is not None:
    # Example to show how easy it is to download ARM data if a username/token are set
    results = act.discovery.download_arm_data(
        username, token, 'bnfmetM1.b1', '2024-10-24', '2024-10-31'
    )
    ds_arm = act.io.arm.read_arm_netcdf(results)

    # results = act.discovery.download_arm_data(
    #    username, token, 'bnfaoso3M1.b1', '2024-10-24', '2024-10-31'
    # )
    # ds_o3 = act.io.arm.read_arm_netcdf(results)

    results = act.discovery.download_arm_data(
        username, token, 'bnfaossmpsM1.b1', '2024-10-24', '2024-10-31'
    )
    ds_smps = act.io.arm.read_arm_netcdf(results)

    display = act.plotting.TimeSeriesDisplay(
        {'ASOS': ds_asos, 'ARM': ds_arm, 'EPA': ds_airnow, 'NOAA': ds_noaa},
        figsize=(12, 10),
        subplot_shape=(3,),
    )
    title = 'Comparison of ARM MET, NOAA Courtland, and Haleyville ASOS Station'
    display.plot('tmpf', dsname='ASOS', label='ASOS', subplot_index=(0,))
    display.plot('Temperature', dsname='NOAA', label='NOAA', subplot_index=(0,))
    display.plot('temp_mean', dsname='ARM', label='ARM', subplot_index=(0,), set_title=title)
    display.day_night_background(dsname='ARM', subplot_index=(0,))

    title = 'Comparison of ARM and EPA Ozone Measurements'
    display.plot('OZONE_sites_0', dsname='EPA', label='EPA ' + sites[0], subplot_index=(1,))
    display.plot('OZONE_sites_1', dsname='EPA', label='EPA2' + sites[1], subplot_index=(1,))
    display.plot(
        'OZONE_sites_2', dsname='EPA', label='EPA3' + sites[2], subplot_index=(1,), set_title=title
    )
    display.day_night_background(dsname='ARM', subplot_index=(1,))

    title = 'ARM SMPS Concentrations and EPA PM2.5'
    display.plot('PM2.5_sites_0', dsname='EPA', label='EPA ' + sites[0], subplot_index=(2,))
    display.plot('PM2.5_sites_1', dsname='EPA', label='EPA ' + sites[1], subplot_index=(2,))
    display.plot(
        'PM2.5_sites_2', dsname='EPA', label='EPA ' + sites[2], subplot_index=(2,), set_title=title
    )
    ax2 = display.axes[2].twinx()
    ax2.plot(ds_smps['time'], ds_smps['total_N_conc'], label='ARM SMPS', color='purple')
    display.day_night_background(dsname='ARM', subplot_index=(2,))
    plt.legend()
    plt.show()
