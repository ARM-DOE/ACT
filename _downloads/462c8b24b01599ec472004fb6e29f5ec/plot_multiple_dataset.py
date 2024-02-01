"""
Plot multiple datasets
----------------------

This is an example of how to download and
plot multiple datasets at a time.

"""

import os

from arm_test_data import DATASETS
import matplotlib.pyplot as plt

import act

# Place your username and token here
username = os.getenv('ARM_USERNAME')
token = os.getenv('ARM_PASSWORD')

# Get data from the web service if username and token are available
# if not, use test data
if username is None or token is None or len(username) == 0 or len(token) == 0:
    filename_ceil = DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc')
    ceil_ds = act.io.arm.read_arm_netcdf(filename_ceil)
    filename_met = DATASETS.fetch('sgpmetE13.b1.20190101.000000.cdf')
    met_ds = act.io.arm.read_arm_netcdf(filename_met)
else:
    # Download and read data
    results = act.discovery.download_arm_data(
        username, token, 'sgpceilC1.b1', '2022-01-01', '2022-01-07'
    )
    ceil_ds = act.io.arm.read_arm_netcdf(results)
    results = act.discovery.download_arm_data(
        username, token, 'sgpmetE13.b1', '2022-01-01', '2022-01-07'
    )
    met_ds = act.io.arm.read_arm_netcdf(results)

# Read in CEIL data and correct it
ceil_ds = act.corrections.ceil.correct_ceil(ceil_ds, -9999.0)


# You can use tuples if the datasets in the tuple contain a
# datastream attribute. This is required in all ARM datasets.
display = act.plotting.TimeSeriesDisplay((ceil_ds, met_ds), subplot_shape=(2,), figsize=(15, 10))
display.plot('backscatter', 'sgpceilC1.b1', subplot_index=(0,))
display.plot('temp_mean', 'sgpmetE13.b1', subplot_index=(1,))
display.day_night_background('sgpmetE13.b1', subplot_index=(1,))
plt.show()

# You can also use a dictionary so that you can customize
# your datastream names to something that may be more useful.
display = act.plotting.TimeSeriesDisplay(
    {'ceiliometer': ceil_ds, 'met': met_ds}, subplot_shape=(2,), figsize=(15, 10)
)
display.plot('backscatter', 'ceiliometer', subplot_index=(0,))
display.plot('temp_mean', 'met', subplot_index=(1,))
display.day_night_background('met', subplot_index=(1,))
plt.show()

ceil_ds.close()
met_ds.close()
