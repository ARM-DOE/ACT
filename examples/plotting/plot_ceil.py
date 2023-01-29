"""
Simple plot of 2D data
----------------------

This is an example of how to download and
plot ceiliometer data from the SGP site
over Oklahoma.

"""

import os
import matplotlib.pyplot as plt
import act

# Place your username and token here
username = os.getenv('ARM_USERNAME')
token = os.getenv('ARM_PASSWORD')

# If the username and token are not set, use the existing sample file
if username is None or token is None or len(username) == 0 or len(token) == 0:
    ceil_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_CEIL1, engine='netcdf4')
else:
    # Example to show how easy it is to download ARM data if a username/token are set
    results = act.discovery.download_data(username, token, 'sgpceilC1.b1', '2022-01-14', '2022-01-19')
    ceil_ds = act.io.armfiles.read_netcdf(results)

# Adjust ceilometer data for plotting
ceil_ds = act.corrections.ceil.correct_ceil(ceil_ds, -9999.0)

# Plot up ceilometer backscatter using HomeyerRainbow cb friendly colormap
# The same could be done with keyword 'cmap='act_HomeyRainbow'
display = act.plotting.TimeSeriesDisplay(ceil_ds, subplot_shape=(1,), figsize=(15, 5))
display.plot('backscatter', subplot_index=(0,), cb_friendly=True)
plt.show()
