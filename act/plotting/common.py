"""
Functions common between plotting modules.

"""

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def parse_ax(ax):
    """
    Parses the given matplotlib axis.

    Parameters
    ----------
    ax : matplotlib axis
        The matplotlib axis to be parsed. Set to none to get the current axis.

    Returns
    -------
    ax : matplotlib axis
        The target matplotlib axis.

    """
    if ax is None:
        ax = plt.gca()
    return ax


def parse_ax_fig(ax, fig):
    """
    Parses the given matplotlib axis and figure.

    Parameters
    ----------
    ax : matplotlib axis
        The matplotlib axis to be parsed. Set to None to get the current axis.
    fig : matplotlib fig
        The matplotlib figure to be parsed.
        Set to None to get the current figure.

    Returns
    -------
    ax : matplotlib axis
        The target matplotlib axis.
    fig : matplotlib figure
        The target matplotlib figure.

    """
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    return ax, fig


def get_date_format(days, day_to_hour_threshold=3):
    """
    Returns the DateFormatter object to use for the given number of days.

    Parameters
    ----------
    days : float
        The number of days we are plotting.
    day_to_hour_threshold : float
        If the dataset is under this threshold long, format the x axis ticks
        by hour instead of date.

    Returns
    -------
    myFmt : matplotlib DateFormatter
        The DateFormatter object to use for the plot.

    """
    # Set format for time axis if needed
    if days > day_to_hour_threshold:
        myFmt = DateFormatter('%b %d')
    else:
        myFmt = DateFormatter('%H:%M')

    return myFmt
