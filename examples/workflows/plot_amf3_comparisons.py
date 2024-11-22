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

# Get Surface Meteorology data from the ASOS stations
station = '1M4'
time_window = [datetime(2024, 10, 19), datetime(2024, 10, 24)]
ds_asos = act.discovery.get_asos_data(time_window, station=station, regions='AL')[station]
ds_asos = ds_asos.where(~np.isnan(ds_asos.tmpf), drop=True)
ds_asos['tmpf'].attrs['units'] = 'degF'
ds_asos.utils.change_units(variables='tmpf', desired_unit='degC', verbose=True)

# Pull EPA data from AirNow
# You need an account and token from https://docs.airnowapi.org/ first
airnow_token = os.getenv('AIRNOW_API')
if airnow_token is not None and len(airnow_token) > 0:
    latlon = '-87.453,34.179,-86.477,34.787'
    ds_airnow = act.discovery.get_airnow_bounded_obs(
        airnow_token, '2024-10-19T00', '2024-10-24T23', latlon, 'OZONE,PM25', data_type='B'
    )
    ds_airnow = act.utils.convert_2d_to_1d(ds_airnow, parse='sites')
    sites = ds_airnow['sites'].values
    airnow = True

# Get NOAA PSL Data from Courtland
results = act.discovery.download_noaa_psl_data(
    site='ctd', instrument='Temp/RH', startdate='20241019', enddate='20241024'
)
ds_noaa = act.io.read_psl_surface_met(results)

# Place your username and token here
username = os.getenv('ARM_USERNAME')
token = os.getenv('ARM_PASSWORD')

# Download ARM data for the MET, OZONE, and SMPS
if username is not None and token is not None and len(username) > 1:
    # Example to show how easy it is to download ARM data if a username/token are set
    results = act.discovery.download_arm_data(
        username, token, 'bnfmetM1.b1', '2024-10-19', '2024-10-24'
    )
    ds_arm = act.io.arm.read_arm_netcdf(results)

    results = act.discovery.download_arm_data(
        username, token, 'bnfaoso3M1.b1', '2024-10-19', '2024-10-24'
    )
    ds_o3 = act.io.arm.read_arm_netcdf(results, cleanup_qc=True)
    ds_o3.qcfilter.datafilter('o3', rm_assessments=['Suspect', 'Bad'], del_qc_var=False)

    results = act.discovery.download_arm_data(
        username, token, 'bnfaossmpsM1.b1', '2024-10-19', '2024-10-24'
    )
    ds_smps = act.io.arm.read_arm_netcdf(results)

    # Set up display and plot all the data
    display = act.plotting.TimeSeriesDisplay(
        {'ASOS': ds_asos, 'ARM': ds_arm, 'EPA': ds_airnow, 'NOAA': ds_noaa, 'ARM_O3': ds_o3},
        figsize=(12, 10),
        subplot_shape=(3,),
    )
    # Plot surface temperature from ASOS, NOAA, and ARM sites
    title = 'Comparison of ARM MET, NOAA Courtland, and Haleyville ASOS Station'
    display.plot('tmpf', dsname='ASOS', label='ASOS', subplot_index=(0,))
    display.plot('Temperature', dsname='NOAA', label='NOAA', subplot_index=(0,))
    display.plot('temp_mean', dsname='ARM', label='ARM', subplot_index=(0,), set_title=title)
    display.day_night_background(dsname='ARM', subplot_index=(0,))

    # Plot ARM and EPA Ozone data
    title = 'Comparison of ARM and EPA Ozone Measurements'
    display.plot('o3', dsname='ARM_O3', label='ARM', subplot_index=(1,))
    if airnow:
        display.plot('OZONE_sites_1', dsname='EPA', label='EPA' + sites[1], subplot_index=(1,))
        display.plot(
            'OZONE_sites_2',
            dsname='EPA',
            label='EPA' + sites[2],
            subplot_index=(1,),
            set_title=title,
        )
    display.set_yrng([0, 70], subplot_index=(1,))
    display.day_night_background(dsname='ARM', subplot_index=(1,))

    # Plot ARM SMPS Concentrations and EPA PM2.5 data on different axes
    title = 'ARM SMPS Concentrations and EPA PM2.5'
    if airnow:
        display.plot('PM2.5_sites_0', dsname='EPA', label='EPA ' + sites[0], subplot_index=(2,))
        display.plot(
            'PM2.5_sites_2',
            dsname='EPA',
            label='EPA ' + sites[2],
            subplot_index=(2,),
            set_title=title,
        )
    display.set_yrng([0, 25], subplot_index=(2,))
    ax2 = display.axes[2].twinx()
    ax2.plot(ds_smps['time'], ds_smps['total_N_conc'], label='ARM SMPS', color='purple')
    ax2.set_ylabel('ARM SMPS (' + ds_smps['total_N_conc'].attrs['units'] + ')')
    ax2.set_ylim([0, 7000])
    ax2.legend(loc=1)
    display.day_night_background(dsname='ARM', subplot_index=(2,))

    # Set legends
    for ax in display.axes:
        ax.legend(loc=2)

    plt.show()
else:
    pass
