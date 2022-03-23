"""
Plot winds and relative humidity from sounding data
---------------------------------------------------

This is an example of how to display wind rose and barb timeseries
from multiple days worth of sounding data.

"""

from matplotlib import pyplot as plt

import act

sonde_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_TWP_SONDE_WILDCARD)

BarbDisplay = act.plotting.TimeSeriesDisplay({'sonde_darwin': sonde_ds}, figsize=(10, 5))
BarbDisplay.plot_time_height_xsection_from_1d_data(
    'rh', 'pres', cmap='YlGn', vmin=0, vmax=100, num_time_periods=25
)
BarbDisplay.plot_barbs_from_spd_dir('deg', 'wspd', 'pres', num_barbs_x=20)
plt.show()
