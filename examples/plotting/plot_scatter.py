"""
Compare Aircraft Airspeeds
--------------------------

Compare Aircraft Airspeeds via the ComparisonDisplay
Scatter Plot

Written: Joe O'Brien

"""

import act

from act.io.icartt import read_icartt

# Call the read_icartt function, which supports input
# for ICARTT (v2.0) formatted files.
# Example file is ARM Aerial Facility Navigation Data
ds = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)

# Create a ComparisonDisplay object to compare fields
display = act.plotting.ComparisonDisplay(ds)

# Compare aircraft ground speed with indicated airspeed
display.scatter('true_airspeed',
                'ground_speed',
                m_field='ambient_temp',
                marker='x',
                cbar_label='Ambient Temperature ($^\circ$C)'
                )

# Set the range of the field on the x-axis
display.set_xrng((40, 140))
display.set_yrng((40, 140))
# Display the 1:1 ratio line
display.set_ratio_line()
