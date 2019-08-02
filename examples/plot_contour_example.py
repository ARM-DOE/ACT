"""
=======================================
Example for plotting up a contour plot
=======================================

This is an example of how to prepare 
and plot data for a contour plot

.. image:: ../../plot_contour_example.png
"""

import act
import glob
import matplotlib.pyplot as plt

files = glob.glob(act.tests.sample_files.EXAMPLE_MET_CONTOUR)
time = '2019-05-08T04:00:00.000000000'
data = {}
fields = {}
for f in files:
    obj = act.io.armfiles.read_netcdf(f)
    data.update({f: obj})
    fields.update({f: ['lon','lat','temp_mean']})

display = act.plotting.ContourDisplay(data, figsize=(8,8))
display.create_contour(fields=fields, time=time, levels=50)
plt.show()
