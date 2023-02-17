"""
Multidimensional cross sections
-------------------------------

In this example, the VISST data are used to
plot up cross-sectional slices through the
multi-dimensional dataset
"""

from datetime import datetime

import matplotlib.pyplot as plt
import xarray as xr

import act

my_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_VISST)

# Cross section display requires that the variable being plotted be reduced to two
# Dimensions whose coordinates can be specified by variables in the file
disp = act.plotting.XSectionDisplay(my_ds, figsize=(20, 8), subplot_shape=(2, 2))
disp.plot_xsection_map(
    None,
    'ir_temperature',
    x='longitude',
    y='latitude',
    cmap='Greys',
    vmin=200,
    vmax=320,
    subplot_index=(0, 0),
)
disp.plot_xsection_map(
    None,
    'ir_temperature',
    x='longitude',
    y='latitude',
    cmap='Greys',
    vmin=200,
    vmax=320,
    subplot_index=(1, 0),
)
disp.plot_xsection_map(
    None,
    'ir_temperature',
    x='longitude',
    y='latitude',
    cmap='Greys',
    vmin=200,
    vmax=320,
    subplot_index=(0, 1),
)
disp.plot_xsection_map(
    None,
    'ir_temperature',
    x='longitude',
    y='latitude',
    cmap='Greys',
    vmin=200,
    vmax=320,
    subplot_index=(1, 1),
)

plt.show()
my_ds.close()
