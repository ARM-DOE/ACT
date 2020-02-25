"""
===========================
act.plotting (act.plotting)
===========================

.. currentmodule:: act.plotting

This module contains classes for displaying data.
:func:`act.plotting.Display` is the
base class on which all other Display classes are inherited from. If you are making
a new Display object, please make it inherited from this class.

| :func:`act.plotting.ContourDisplay` handles the plotting of contour plots.
| :func:`act.plotting.GeographicPlotDisplay` handles the plotting of lat-lon plots.
| :func:`act.plotting.HistogramDisplay` handles the plotting of histogram plots.
| :func:`act.plotting.SkewTDisplay` handles the plotting of Skew-T diagrams.
| :func:`act.plotting.TimeSeriesDisplay` handles the plotting of timeseries.
| :func:`act.plotting.WindRoseDisplay` handles the plotting of wind rose plots.
| :func:`act.plotting.XSectionDisplay` handles the plotting of cross sections.

.. autosummary::
    :toctree: generated/

    plot.Display
    ContourDisplay.ContourDisplay
    ContourDisplay.ContourDisplay
    GeoDisplay.GeographicPlotDisplay
    HistogramDisplay.HistogramDisplay
    SkewTDisplay.SkewTDisplay
    TimeSeriesDisplay.TimeSeriesDisplay
    WindRoseDisplay.WindRoseDisplay
    XSectionDisplay.XSectionDisplay
    common.parse_ax
    common.parse_ax_fig
    common.get_date_format
    act_cmap
"""

from .plot import Display
from .TimeSeriesDisplay import TimeSeriesDisplay
from .ContourDisplay import ContourDisplay
from .WindRoseDisplay import WindRoseDisplay
from .SkewTDisplay import SkewTDisplay
from .XSectionDisplay import XSectionDisplay
from .GeoDisplay import GeographicPlotDisplay
from .HistogramDisplay import HistogramDisplay
from . import common
from . import act_cmap
