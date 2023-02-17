"""
Calculate and plot wind rose plots separated by day.
-----------------------------------------------------

Example of how to read in MET data and plot histograms
of wind speed and temperature grouped by day.

Author: Adam Theisen
"""

import matplotlib.pyplot as plt
import numpy as np
import act

# Read in the sample MET data
obj = act.io.armfiles.read_netcdf(act.tests.EXAMPLE_MET_WILDCARD)

# Create Plot Display
display = act.plotting.WindRoseDisplay(obj, figsize=(15, 15), subplot_shape=(3, 3))
groupby = display.group_by('day')
groupby.plot_group('plot_data', None, dir_field='wdir_vec_mean', spd_field='wspd_vec_mean',
                   data_field='temp_mean', num_dirs=12, plot_type='line')

plt.show()

obj.close()
