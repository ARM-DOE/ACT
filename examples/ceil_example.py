"""
======================================
Example for looking at ceilometer data
======================================

This is an example of how to download and
plot ceiliometer data from the SGP site
over Oklahoma.

.. image:: ../../plot_ceil_example.png
"""

import matplotlib.pyplot as plt

import act

# Place your username and token here
username = ''
token = ''

act.discovery.download_data(username, token, 'sgpceilC1.b1', '2017-01-14', '2017-01-19')

ceil_ds = act.io.armfiles.read_netcdf('sgpceilC1.b1/*')
print(ceil_ds)
ceil_ds = act.corrections.ceil.correct_ceil(ceil_ds, -9999.0)
display = act.plotting.TimeSeriesDisplay(ceil_ds, subplot_shape=(1,), figsize=(15, 5))
display.plot('backscatter', subplot_index=(0,))
plt.show()
