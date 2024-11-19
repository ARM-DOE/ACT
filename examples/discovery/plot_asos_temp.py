"""
Query and plot ASOS data
===========================================

This example shows how to plot timeseries of ASOS data from
Chicago O'Hare airport.

"""
from datetime import datetime
import matplotlib.pyplot as plt
import act

time_window = [datetime(2020, 2, 4, 2, 0), datetime(2020, 2, 10, 10, 0)]
my_asoses = act.discovery.get_asos_data(time_window, station='ORD', regions='IL')

display = act.plotting.TimeSeriesDisplay(my_asoses['ORD'], subplot_shape=(2,), figsize=(15, 10))
display.plot('temp', subplot_index=(0,))
display.plot_barbs_from_u_v(u_field='u', v_field='v', subplot_index=(1,))
display.axes[1].set_ylim([0, 2])
plt.show()
