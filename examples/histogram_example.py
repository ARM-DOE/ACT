"""
Example for plotting a density plot of CO2 concentration versus time
--------------------------------------------------------------------
"""
import act
import datetime
import numpy as np
import matplotlib.pyplot as plt

my_ds = act.io.csvfiles.read_csv("AMF_US-ARM_BASE_HH_8-5.csv", skiprows=2)
timestamp_strs = [str(x) for x in my_ds["TIMESTAMP_START"].values]
my_ds["time"] = np.array([datetime.datetime.strptime(x, "%Y%m%d%H%M") for x in timestamp_strs])
my_ds["year"] = np.array([x.astype('datetime64[Y]').astype(int)+1970 for x in my_ds["time"].values])

Histogram = act.plotting.HistogramDisplay(my_ds)
Histogram.plot_density("year", "CO2_1_1_1", y_bins=np.linspace(350, 450, 40),
                       x_bins=np.arange(2003, 2020, 1), density=True)
plt.show()
print(my_ds)