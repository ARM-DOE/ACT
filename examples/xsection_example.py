"""
Example for plotting multidimensional cross sections
====================================================

In this example, the VISST
"""

from datetime import datetime

import matplotlib.pyplot as plt
import xarray as xr

import act

my_ds = act.io.armfiles.read_netcdf('twpvisst*')

# Cross section display requires that the variable being plotted be reduced to two
# Dimensions whose coordinates can be specified by variables in the file
print(my_ds)
disp = act.plotting.XSectionDisplay(my_ds, subplot_shape=(2, 2))
disp.plot_xsection_map(
    None,
    'ir_temperature',
    x='longitude',
    y='latitude',
    sel_kwargs={'time': datetime(2005, 7, 5, 1, 45, 00)},
    isel_kwargs={'scn_type': 0},
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
    sel_kwargs={'time': datetime(2005, 7, 5, 2, 45, 00)},
    isel_kwargs={'scn_type': 0},
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
    sel_kwargs={'time': datetime(2005, 7, 5, 3, 25, 00)},
    isel_kwargs={'scn_type': 0},
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
    sel_kwargs={'time': datetime(2005, 7, 5, 3, 55, 00)},
    isel_kwargs={'scn_type': 0},
    cmap='Greys',
    vmin=200,
    vmax=320,
    subplot_index=(1, 1),
)

plt.show()
my_ds.close()
