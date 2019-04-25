"""
Example on how to plot a Skew-T plot of a sounding
--------------------------------------------------

This example shows how to make a Skew-T plot from a sounding.
"""
import act
import numpy as np

from matplotlib import pyplot as plt

sonde_ds = act.io.armfiles.read_netcdf(
    act.tests.sample_files.EXAMPLE_SONDE1)

skewt = act.plotting.SkewTDisplay(sonde_ds)

skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')

plt.show(skewt.fig)

sonde_ds.close()
