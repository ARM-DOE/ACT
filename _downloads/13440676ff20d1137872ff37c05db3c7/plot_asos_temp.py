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
station = 'KORD'
my_asoses = act.discovery.get_asos(time_window, station='ORD')

my_disp = act.plotting.TimeSeriesDisplay(my_asoses['ORD'], subplot_shape=(2,), figsize=(15, 10))
my_disp.plot('temp', subplot_index=(0,))
my_disp.plot_barbs_from_u_v(u_field='u', v_field='v', subplot_index=(1,))
my_disp.axes[1].set_ylim([0, 2])
plt.show()
