"""
Compare Aircraft Airspeeds
--------------------------

Compare Aircraft Airspeeds via the DistributionDisplay
Scatter Plot

Written: Joe O'Brien

"""

import act
import numpy as np

from scipy.stats.mstats import pearsonr
from act.io.icartt import read_icartt

# Call the read_icartt function, which supports input
# for ICARTT (v2.0) formatted files.
# Example file is ARM Aerial Facility Navigation Data
ds = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)

# Create a DistributionDisplay object to compare fields
display = act.plotting.DistributionDisplay(ds)

# Compare aircraft ground speed with indicated airspeed
display.plot_scatter('true_airspeed',
                     'ground_speed',
                     m_field='ambient_temp',
                     marker='x',
                     cbar_label='Ambient Temperature ($^\circ$C)'
                     )

# Set the range of the field on the x-axis
display.set_xrng((40, 140))
display.set_yrng((40, 140))

# Determine the best fit line
z = np.ma.polyfit(ds['true_airspeed'],
                  ds['ground_speed'],
                  1
                  )
p = np.poly1d(z)

# Plot the best fit line
display.axes[0].plot(ds['true_airspeed'],
                     p(ds['true_airspeed']),
                     'r',
                     linewidth=2
                     )

# Display the line equation
display.axes[0].text(45,
                     135,
                     "y = %.3fx + (%.3f)" % (z[0], z[1]),
                     color='r',
                     fontsize=12
                     )

# Calculate Pearson Correlation Coefficient
cc_conc = pearsonr(ds['true_airspeed'],
                   ds['ground_speed']
                   )

# Display the Pearson CC
display.axes[0].text(45,
                     130,
                     "Pearson CC: %.2f" % (cc_conc[0]),
                     fontsize=12
                     )

# Display the total number of samples
display.axes[0].text(45,
                     125,
                     "N = %.0f" % (ds['true_airspeed'].data.shape[0]),
                     fontsize=12
                     )

# Display the 1:1 ratio line
display.set_ratio_line()
