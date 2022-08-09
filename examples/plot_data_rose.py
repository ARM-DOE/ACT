"""
Data rose plot
--------------

This is an example of how to display a data rose.
As can be seen in the final plot, there are two major
bullseyes of data, one around 0ºC to the Northeast and
another around 15ºC to the South. This tells us that we
get lower temperatures when winds are out of the N/NE as
would be expected at this location.  This can be extended
to easily review other types of data as well like aerosols
and fluxes.

"""

import numpy as np
from matplotlib import pyplot as plt
import act

# Read in some data that has wind directions
obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET_WILDCARD)

# Create the display and plot up a contour of temp_mean
display = act.plotting.WindRoseDisplay(obj)
display.plot_data('wdir_vec_mean', 'wspd_vec_mean', 'temp_mean',
                  num_dirs=30, plot_type='Contour', clevels=50)
plt.show()
