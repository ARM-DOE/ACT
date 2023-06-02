"""
Investigate Temperature Quantiles
---------------------------------

Investigate temperature quantiles using
ComparisonDisplay Violin plot

Written: Joe O'Brien

"""
import matplotlib.pyplot as plt

import act

from act.io.icartt import read_icartt

# Call the read_icartt function, which supports input
# for ICARTT (v2.0) formatted files.
# Example file is ARM Aerial Facility Navigation Data
ds = read_icartt(act.tests.EXAMPLE_AAF_ICARTT)

# Create a ComparisonDisplay object to compare fields
display = act.plotting.ComparisonDisplay(ds)

# Create a figure and axe to hold this display
#fig, axe = plt.subplots(figsize=[12, 8])

# Compare aircraft ground speed with ambient temperature
display.violin('ambient_temp',
                positions=[1.0],
                vert = True,
                showmeans = True,
                showmedians = True,
                showextrema = True,
                )

display.violin('total_temp',
                positions=[2.0],
                vert = True,
                showmeans = True,
                showmedians = True,
                showextrema = True,
                )

# Assign display object to figure axe


#display.scatter('true_airspeed', 
#                'ground_speed',
#                ratio_line=True
#                )
# Set the range of the field on the x-axis
#display.set_xrng((40, 140))
#display.set_yrng((40, 140))
# Display the 1:1 ratio line
#display.set_ratio_line()
