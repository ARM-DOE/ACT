"""
Plot winds and relative humidity from sounding data
---------------------------------------------------

This is an example of how to display wind rose and barb timeseries
from multiple days worth of sounding data.

"""

from arm_test_data import DATASETS
from matplotlib import pyplot as plt

import act

# Read in sonde files
twp_sonde_wildcard_list = [
    'twpsondewnpnC3.b1.20060119.050300.custom.cdf',
    'twpsondewnpnC3.b1.20060119.112000.custom.cdf',
    'twpsondewnpnC3.b1.20060119.163300.custom.cdf',
    'twpsondewnpnC3.b1.20060119.231600.custom.cdf',
    'twpsondewnpnC3.b1.20060120.043800.custom.cdf',
    'twpsondewnpnC3.b1.20060120.111900.custom.cdf',
    'twpsondewnpnC3.b1.20060120.170800.custom.cdf',
    'twpsondewnpnC3.b1.20060120.231500.custom.cdf',
    'twpsondewnpnC3.b1.20060121.051500.custom.cdf',
    'twpsondewnpnC3.b1.20060121.111600.custom.cdf',
    'twpsondewnpnC3.b1.20060121.171600.custom.cdf',
    'twpsondewnpnC3.b1.20060121.231600.custom.cdf',
    'twpsondewnpnC3.b1.20060122.052600.custom.cdf',
    'twpsondewnpnC3.b1.20060122.111500.custom.cdf',
    'twpsondewnpnC3.b1.20060122.171800.custom.cdf',
    'twpsondewnpnC3.b1.20060122.232600.custom.cdf',
    'twpsondewnpnC3.b1.20060123.052500.custom.cdf',
    'twpsondewnpnC3.b1.20060123.111700.custom.cdf',
    'twpsondewnpnC3.b1.20060123.171600.custom.cdf',
    'twpsondewnpnC3.b1.20060123.231500.custom.cdf',
    'twpsondewnpnC3.b1.20060124.051500.custom.cdf',
    'twpsondewnpnC3.b1.20060124.111800.custom.cdf',
    'twpsondewnpnC3.b1.20060124.171700.custom.cdf',
    'twpsondewnpnC3.b1.20060124.231500.custom.cdf',
]
sonde_filenames = [DATASETS.fetch(file) for file in twp_sonde_wildcard_list]
sonde_ds = act.io.arm.read_arm_netcdf(sonde_filenames)

BarbDisplay = act.plotting.TimeSeriesDisplay({'sonde_darwin': sonde_ds}, figsize=(10, 5))
BarbDisplay.plot_time_height_xsection_from_1d_data(
    'rh', 'pres', cmap='YlGn', vmin=0, vmax=100, num_time_periods=25
)
BarbDisplay.plot_barbs_from_spd_dir('wspd', 'deg', 'pres', num_barbs_x=20)
plt.show()
