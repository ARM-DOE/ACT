"""
==========================================
Example for calculating a weighted average
==========================================

This is an example of how to calculate a
weighted average from the MET TBRG, ORG
and PWD.  This also calculates the 
accumulated precipitation and displays it

.. image:: ../../weighted_average_example.png
"""

import act
import matplotlib.pyplot as plt
import xarray as xr

# Specify dictionary of datastreams, variables, and weights
# Note, all weights should add up to 1.
cf_ds = {'sgpmetE13.b1': {'variable': ['tbrg_precip_total', 'org_precip_rate_mean',
                                       'pwd_precip_rate_mean_1min'],
                          'weight': [0.8, 0.15, 0.05]}}

# Other way to define cf_ds
# cf_ds = {'sgpmetE13.b1': {'variable': ['tbrg_precip_total'], 'weight': [0.5]},
#         'sgpmetE13.b1': {'variable': ['org_precip_rate_mean'], 'weight': [0.25]},
#         'sgpmetE13.b1': {'variable': ['pwd_precip_rate_mean_1min'], 'weight': [0.25]}
#        }

ds = {}
new = {}
out_units = 'mm/hr'
for d in cf_ds:
    obj = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_MET_WILDCARD)
    # Loop through each variable and add to data list
    new_da = []
    for v in cf_ds[d]['variable']:
        da = obj[v]
        # Accumulate precip variables in new object i
        obj = act.utils.data_utils.accumulate_precip(obj, v)

        # Convert units and add to dataarray list
        units = da.attrs['units']
        if units == 'mm':
            da.attrs['units'] = 'mm/min'
        da.values = act.utils.data_utils.convert_units(da.values, da.attrs['units'], out_units)
        da = da.resample(time='1min').mean()
        new_da.append(da)

    # Depending on number of variables for each datastream, merge or create dataset
    if len(new_da) > 1:
        new_da = xr.merge(new_da)
    else:
        new_da = new_da[0].to_dataset()

    # Add to dictionary for the weighting
    cf_ds[d]['object'] = new_da

    # Add object to dictionary for plotting
    new[d] = obj

# Calculate weighted averages using the dict defined above
data = act.utils.data_utils.ts_weighted_average(cf_ds)

# Add weighted mean to plotting object and calculate accumulation
new['weighted'] = data.to_dataset(name='weighted_mean')
new['weighted']['weighted_mean'].attrs['units'] = 'mm/hr'
new['weighted'] = act.utils.data_utils.accumulate_precip(new['weighted'], 'weighted_mean')

# Plot the accumulations
display = act.plotting.TimeSeriesDisplay(new, figsize=(12, 8), subplot_shape=(1,))
display.plot('tbrg_precip_total_accumulated', dsname='sgpmetE13.b1', color='b', label='TBRG, 0.8')
display.plot('org_precip_rate_mean_accumulated', dsname='sgpmetE13.b1', color='g', label='ORG 0.15')
display.plot('pwd_precip_rate_mean_1min_accumulated', dsname='sgpmetE13.b1', color='y', label='PWD 0.05')
display.plot('weighted_mean_accumulated', dsname='weighted', color='k', label='Weighted Avg')
display.day_night_background('sgpmetE13.b1')
display.axes[0].legend()
plt.show()
