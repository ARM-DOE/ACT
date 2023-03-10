"""
Plot a timeseries of sounding data
----------------------------------------------------

This is a simple example for how to plot a timeseries of sounding
data from the ARM SGP site.

Author: Robert Jackson
"""

from matplotlib import pyplot as plt

import act

files = act.tests.sample_files.EXAMPLE_MET_WILDCARD
met_ds = act.io.armfiles.read_netcdf(files)
print(met_ds)
met_temp = met_ds.temp_mean
met_rh = met_ds.rh_mean
met_lcl = (20.0 + met_temp / 5.0) * (100.0 - met_rh) / 1000.0
met_ds['met_lcl'] = met_lcl * 1000.0
met_ds['met_lcl'].attrs['units'] = 'm'
met_ds['met_lcl'].attrs['long_name'] = 'LCL Calculated from SGP MET E13'

# Plot data
display = act.plotting.TimeSeriesDisplay(met_ds)
display.add_subplots((3,), figsize=(15, 10))
display.plot('wspd_vec_mean', subplot_index=(0,))
display.plot('temp_mean', subplot_index=(1,))
display.plot('rh_mean', subplot_index=(2,))
plt.show()
