"""
Plot ARM AAF Flight Path
--------------------------------

Plot the ARM AAF flight path using the GeographicPlotDisplay

Author: Joe O'Brien

"""
import matplotlib.pyplot as plt

import act
from act.io.icartt import read_icartt

# Call the read_icartt function, which supports input
# for ICARTT (v2.0) formatted files.
# Example file is ARM Aerial Facility Navigation Data
ds = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)

# Use GeographicPlotDisplay for referencing.
# NOTE: Cartopy is needed!
display = act.plotting.GeographicPlotDisplay(ds, figsize=(12, 10))

# Plot the ARM AAF flight track with respect to Pressure Altitude
display.geoplot('press_alt', lat_field='lat', lon_field='lon')

# Display the plot
plt.show()
