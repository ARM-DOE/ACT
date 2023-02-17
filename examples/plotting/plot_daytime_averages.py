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
obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET_WILDCARD)

# Add the solar variable, including dawn/dusk to variable
obj = act.utils.geo_utils.add_solar_variable(obj)

# Using the sun variable, only analyze daytime data
obj = obj.where(obj['sun_variable'] == 1)

# Take daily mean using xarray features
obj = obj.resample(time='1d', skipna=True, keep_attrs=True).mean()

# Creat Plot Display
display = act.plotting.TimeSeriesDisplay(obj, figsize=(15, 10))
display.plot('temp_mean', linestyle='solid')
display.day_night_background()
plt.show()

obj.close()
