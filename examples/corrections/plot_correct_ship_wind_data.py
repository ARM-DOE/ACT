"""
Correct wind data for ship motion
---------------------------------------------------

This example shows how to calculate course and speed
over ground of the ship and use it to correct the
wind speed and direction data.
"""
import xarray as xr

import act

# Read in the navigation data, mainly for the lat/lon
nav_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_NAV)

# Calculate course and speed over ground from the NAV
# lat and lon data
nav_ds = act.utils.ship_utils.calc_cog_sog(nav_ds)

# Read in the data containing the wind speed and direction
aosmet_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_AOSMET)

# Merge the navigation and wind data together
# This have been previously resampled to 1-minute data
ds = xr.merge([nav_ds, aosmet_ds], compat='override')

# Call the correction for the winds.  Note, that this only
# corrects for ship course and speed, not roll and pitch.
ds = act.corrections.ship.correct_wind(ds)

nav_ds.close()
aosmet_ds.close()
ds.close()
