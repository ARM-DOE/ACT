"""
Using ACT for Satellite data
----------------------------

Simple example for working with satellite data.
It is recommended that users explore other
satellite specific libraries suchas SatPy.

Author: Adam Theisen
"""

import act
import matplotlib.pyplot as plt
import numpy as np
from arm_test_data import DATASETS

# Read in VISST Data
files = DATASETS.fetch('enavisstgridm11minnisX1.c1.20230307.000000.cdf')
ds = act.io.read_arm_netcdf(files)

# Plot up the VISST cloud percentage using XSectionDisplay
display = act.plotting.XSectionDisplay(ds, figsize=(10, 8))
display.plot_xsection_map(
    'cloud_percentage', x='longitude', y='latitude', isel_kwargs={'time': 0, 'cld_type': 0}
)
plt.show()

# Download ARM TSI Data
files = DATASETS.fetch('enatsiskycoverC1.b1.20230307.082100.cdf')
ds_tsi = act.io.read_arm_netcdf(files)
ds_tsi = ds_tsi.where(ds_tsi.percent_opaque > 0)
ds_tsi = ds_tsi.resample(time='30min').mean()

# Set coordinates to extra data for ENA
ena_lat = 39.091600
ena_lon = 28.025700

lat = ds['lat'].values
lon = ds['lon'].values

# Find the nearest pixel for the satellite data and extract it
lat_ind = np.argmin(np.abs(lat - ena_lat))
lon_ind = np.argmin(np.abs(lon - ena_lon))

ds_new = ds.isel(lat=lat_ind, lon=lon_ind, cld_type=0)

# Plot the comparison using TimeSeriesDisplay
display = act.plotting.TimeSeriesDisplay({'Satellite': ds_new, 'ARM': ds_tsi}, figsize=(15, 8))
display.plot('cloud_percentage', dsname='Satellite', label='VISST Cloud Percentage')
display.plot('percent_opaque', dsname='ARM', label='ARM TSI Percent Opaque')
display.day_night_background(dsname='ARM')
plt.legend()
plt.show()
