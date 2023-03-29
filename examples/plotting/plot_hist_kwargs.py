"""
Plot a histogram of Met data.
----------------------------------------------------

This is a simple example for how to plot a histogram
of Meteorological data, while using hist_kwargs parameter.

Author: Zachary Sherman
"""

from matplotlib import pyplot as plt
import numpy as np

import act

files = act.tests.sample_files.EXAMPLE_MET1
met_ds = act.io.armfiles.read_netcdf(files)

# Plot data
hist_kwargs = {'range': (-10, 10)}
histdisplay = act.plotting.HistogramDisplay(met_ds)
histdisplay.plot_stacked_bar_graph('temp_mean', bins=np.arange(-40, 40, 5),
                                   hist_kwargs=hist_kwargs)
plt.show()
