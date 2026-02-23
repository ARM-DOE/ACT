"""
Plot a timeseries of sounding data
----------------------------------------------------

This is a simple example for how to plot a timeseries of sounding
data from the ARM SGP site.

Author: Robert Jackson
"""

from arm_test_data import DATASETS
from matplotlib import pyplot as plt

import act

filename_sonde = DATASETS.fetch('sgpsondewnpnC1.b1.20190101.053200.cdf')
sonde_ds = act.io.arm.read_arm_netcdf(filename_sonde)
print(sonde_ds)

# Plot data
display = act.plotting.TimeSeriesDisplay(sonde_ds)
display.add_subplots((3,), figsize=(15, 10))
display.plot('wspd', subplot_index=(0,))
display.plot('tdry', subplot_index=(1,))
display.plot('rh', subplot_index=(2,))
plt.show()
