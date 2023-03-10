"""
Plot a timeseries of sounding data
----------------------------------------------------

This is a simple example for how to plot multiple columns
in a TimeseriesDisplay.

Author: Maxwell Grover
"""

from matplotlib import pyplot as plt

import act

files = act.tests.sample_files.EXAMPLE_MET_WILDCARD
met_ds = act.io.armfiles.read_netcdf(files)


# Plot data
display = act.plotting.TimeSeriesDisplay(met_ds)
display.add_subplots((3, 2), figsize=(15, 10))
display.plot('temp_mean', color='tab:red', subplot_index=(0, 0))
display.plot('rh_mean', color='tab:green', subplot_index=(1, 0))
display.plot('wdir_vec_mean', subplot_index=(2, 0))
display.plot('temp_std', color='tab:red', subplot_index=(0, 1))
display.plot('rh_std', color='tab:green', subplot_index=(1, 1))
display.plot('wdir_vec_std', subplot_index=(2, 1))
plt.show()
