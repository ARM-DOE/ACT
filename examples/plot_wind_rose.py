"""
=====================================================================
Example on how to plot winds and relative humidity from sounding data
=====================================================================

This is an example of how to display wind rose and barb timeseries
from multiple days worth of sounding data.

"""

import act
import numpy as np

from matplotlib import pyplot as plt

sonde_ds = act.io.armfiles.read_netcdf(
    act.tests.sample_files.EXAMPLE_TWP_SONDE_WILDCARD)

WindDisplay = act.plotting.WindRoseDisplay(sonde_ds, figsize=(8, 10))
WindDisplay.plot('deg', 'wspd',
                 spd_bins=np.linspace(0, 25, 5), num_dirs=30,
                 tick_interval=2)
plt.show()

BarbDisplay = act.plotting.TimeSeriesDisplay(
    {'sonde_darwin': sonde_ds}, figsize=(10, 5))
BarbDisplay.plot_time_height_xsection_from_1d_data('rh', 'pres',
                                                   cmap='coolwarm_r',
                                                   vmin=0, vmax=100,
                                                   num_time_periods=25)
BarbDisplay.plot_barbs_from_spd_dir('deg', 'wspd', 'pres',
                                    num_barbs_x=20)
plt.show()
