"""
Calculate and plot daily daytime temperature averages
-----------------------------------------------------

Example of how to read in MET data and plot up daytime
temperature averages using the add_solar_variable function

Author: Adam Theisen
"""

import matplotlib.pyplot as plt

import act

# Read in the sample MET data
ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET_WILDCARD)

# Add the solar variable, including dawn/dusk to variable
ds = act.utils.geo_utils.add_solar_variable(ds)

# Using the sun variable, only analyze daytime data
ds = ds.where(ds['sun_variable'] == 1)

# Take daily mean using xarray features
ds = ds.resample(time='1d', skipna=True, keep_attrs=True).mean()

# Creat Plot Display
display = act.plotting.TimeSeriesDisplay(ds, figsize=(15, 10))
display.plot('temp_mean', linestyle='solid')
display.day_night_background()
plt.show()

ds.close()
