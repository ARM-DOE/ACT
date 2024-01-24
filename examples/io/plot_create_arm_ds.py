"""
Create a dataset to mimic ARM file formats
------------------------------------------
Example shows how to create a dataset from an ARM DOD.
This will enable users to create files that mimic ARM
files, making for easier use across the community.

Author: Adam Theisen

"""

import act

# Create an empty dataset using an ARM DOD
ds = act.io.arm.create_ds_from_arm_dod('vdis.b1', {'time': 1440}, scalar_fill_dim='time')

# Print out the xarray dataset to see that it's empty
print(ds)

# The user could populate this through a number of ways
# and that's best left up to the user on how to do it.
# If one has an existing dataset, a mapping of variable
# names is sometimes the easiest way

# Let's look at some variable attributes
# These can be updated and it would be up to the
# user to ensure these tests are being applied
# and are appropriately set in the cooresponding QC variable
print(ds['num_drops'].attrs)

# Next, let's print out the global attribuets
print(ds.attrs)

# Add additional attributes or append to existing
# if they are needed using a dictionary
atts = {
    'command_line': 'python  plot_create_arm_ds.py',
    'process_version': '1.2.3',
    'history': 'Processed with Jupyter Workbench',
    'random': '1234253sdgfadf',
}
for a in atts:
    if a in ds.attrs:
        ds.attrs[a] += atts[a]
    else:
        ds.attrs[a] = atts[a]
    # Print out the attribute
    print(a, ds.attrs[a])

# Write data out to netcdf
ds.to_netcdf('./sgpvdisX1.b1.20230101.000000.nc')

# If one wants to clean up the dataset to better match CF standards
# the following can be done as well
ds.write.write_netcdf(cf_compliant=True, path='./sgpvdisX1.b1.20230101.000000.cf')
