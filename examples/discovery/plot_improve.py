"""
IMPROVE Data
-----------

This example shows how to get IMPROVE data for the
system located at ARM's Southern Great Plains site.

"""

import act
import matplotlib.pyplot as plt

# Pull the data using the site_id from IMPROVE
# https://views.cira.colostate.edu/adms/Pub/SiteSummary.aspx?dsidse=10001&siidse=244
ds = act.discovery.get_improve_data(site_id='244', start_date='1/1/2023', end_date='12/31/2023')

# Remove all data that's set to the FillValue
ds = ds.where(ds['aluminum_fine'] != ds['aluminum_fine'].attrs['_FillValue'])

display = act.plotting.TimeSeriesDisplay(ds, figsize=(10, 6))
display.plot('aluminum_fine')

# Print out the known problems documented by IMPROVE
print(ds.attrs['site_problems'])

# Write out the data to netCDF and csv
ds.to_netcdf('./sgpimprove.20230101.nc')
ds.to_dataframe().to_csv('sgpimprove.20230101.csv')

plt.show()
