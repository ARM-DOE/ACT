"""
Read and plot a HYSPLIT trajectory file from a HYSPlIT run.
-----------------------------------------------------------

This example shows how to read and plot a backtrajectory calculated by the NOAA
HYSPLIT model over Houston.

Author: Robert Jackson
"""

import act
import matplotlib.pyplot as plt

from act.tests import sample_files

# Load the data
filename = sample_files.EXAMPLE_HYSPLIT
ds = act.io.read_hysplit(filename)

# Use the GeographicPlotDisplay object to make the plot
disp = act.plotting.GeographicPlotDisplay(ds)
disp.geoplot('PRESSURE', cartopy_feature=['STATES', 'OCEAN', 'LAND'])
plt.show()
