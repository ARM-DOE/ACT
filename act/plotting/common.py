#########################
#Common Plotting Methods#
#########################

import matplotlib.pyplot as plt
from matplotlib.dates import (DateFormatter, HourLocator, DayLocator)

def parse_ax(ax):
    """ Parse and return ax parameter. """
    if ax is None:
        ax = plt.gca()
    return ax


def parse_ax_fig(ax, fig):
    """ Parse and return ax and fig parameters. """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    return ax, fig

def get_date_format(days):
    #Set format for time axis if needed
    minorFmt, majorFmt = None, None
    if days > 3:
        myFmt = DateFormatter('%b %d')
        if days < 10:
            minorFmt = HourLocator(interval=6)
            majorFmt = HourLocator(interval=24)
    else:
        myFmt = DateFormatter('%H:%M')
        minorFmt = HourLocator(interval=1)
        majorFmt = HourLocator(interval=3)

    return myFmt
