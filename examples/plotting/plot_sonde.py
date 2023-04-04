"""
Plot a timeseries of sounding data
----------------------------------------------------

This is a simple example for how to plot a timeseries of sounding
data from the ARM SGP site.

Author: Robert Jackson
"""

from matplotlib import pyplot as plt

import act

files = act.tests.sample_files.EXAMPLE_SONDE1
sonde_ds = act.io.armfiles.read_netcdf(files)
print(sonde_ds)

# Plot data
display = act.plotting.TimeSeriesDisplay(sonde_ds)
display.add_subplots((3,), figsize=(15, 10))
display.plot('wspd', subplot_index=(0,))
display.plot('tdry', subplot_index=(1,))
display.plot('rh', subplot_index=(2,))
plt.show()
