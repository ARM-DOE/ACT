"""
Windrose and windbarb timeseries plot
-------------------------------------

This is an example of how to display wind rose and barb timeseries
from multiple days worth of sounding data.

"""

import numpy as np
from matplotlib import pyplot as plt

import act

sonde_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_TWP_SONDE_WILDCARD)

WindDisplay = act.plotting.WindRoseDisplay(sonde_ds, figsize=(8, 10), subplot_shape=(2,))
WindDisplay.plot(
    'deg', 'wspd', spd_bins=np.linspace(0, 25, 5), num_dirs=30, tick_interval=2, subplot_index=(0,)
)

BarbDisplay = act.plotting.TimeSeriesDisplay({'sonde_darwin': sonde_ds}, figsize=(10, 5))
WindDisplay.put_display_in_subplot(BarbDisplay, subplot_index=(1,))
BarbDisplay.plot_time_height_xsection_from_1d_data(
    'rh', 'pres', cmap='coolwarm_r', vmin=0, vmax=100, num_time_periods=25
)

BarbDisplay.plot_barbs_from_spd_dir('wspd', 'deg', 'pres', num_barbs_x=20)
plt.show()
