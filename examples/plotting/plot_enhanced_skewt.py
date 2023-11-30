"""
Enhanced plot of a sounding
---------------------------

This example shows how to make an enhance plot for sounding data
which includes a Skew-T plot, hodograph, and stability indicies.

Author: Adam Theisen

"""

import glob
import metpy
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import act

# Read data
file = sorted(glob.glob(act.tests.sample_files.EXAMPLE_SONDE1))
ds = act.io.arm.read_arm_netcdf(file)

# Plot enhanced Skew-T plot
display = act.plotting.SkewTDisplay(ds)
display.plot_enhanced_skewt(color_field='alt')

ds.close()
plt.show()
