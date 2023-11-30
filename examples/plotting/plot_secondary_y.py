"""
Secondary Y-Axis Plotting
-------------------------

This example shows how to use the new capability
to plot on the secondary y-axis. Previous versions
of ACT only returned one axis object per plot, even
when there was a secondary y-axis.  The new functionality
will return two axis objects per plot for the left and
right y axes.

"""


import act
import matplotlib.pyplot as plt
import xarray as xr

# Read in the data from a MET file
ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET1)

# Plot temperature and relative humidity with RH on the right axis
display = act.plotting.TimeSeriesDisplay(ds, figsize=(10, 6))

# Plot the data and make the y-axes color match the lines
# Note, you need to specify the color for the secondary-y plot
# if you want it different from the primary y-axis
display.plot('temp_mean', match_line_label_color=True)
display.plot('rh_mean', secondary_y=True, color='orange')
display.day_night_background()

# In a slight change, the axes returned as part of the display object
# for TimeSeries and DistributionDisplay now return a 2D array instead
# of a 1D array.   The second dimension is the axes object for the right
# axis which is automatically created.
# It can still be used like before for modifications after ACT plotting

# The left axis will have an index of 0
#               \/
display.axes[0, 0].set_yticks([-5, 0, 5])
display.axes[0, 0].set_yticklabels(["That's cold", "Freezing", "Above Freezing"])

# The right axis will have an index of 1
#               \/
display.axes[0, 1].set_yticks([65, 75, 85])
display.axes[0, 1].set_yticklabels(['Not as humid', 'Slightly Humid', 'Humid'])

plt.tight_layout()
plt.show()
