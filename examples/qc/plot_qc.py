"""
Plotting QC Flags
-----------------

Simple example for cleaning up a dataset and
plotting the data and its QC flags

Author: Adam Theisen
"""


from matplotlib import pyplot as plt

import act

# Read in sample MET data
files = act.tests.sample_files.EXAMPLE_MET1
obj = act.io.armfiles.read_netcdf(files)

# In order to utilize all the ACT QC modules and plot the QC,
# we need to clean up the object to follow CF standards
obj.clean.cleanup()


# Plot data
# Creat Plot Display
variable = 'temp_mean'
display = act.plotting.TimeSeriesDisplay(obj, figsize=(15, 10), subplot_shape=(2,))

# Plot temperature data in top plot
display.plot(variable, subplot_index=(0,))

# Plot QC data
display.qc_flag_block_plot(variable, subplot_index=(1,))
plt.show()
