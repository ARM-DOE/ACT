"""
===========================
act.plotting (act.plotting)
===========================

.. currentmodule:: act.plotting

This module contains procedures for plotting ARM datasets.

.. autosummary::
    :toctree: generated/

    common.parse_ax
    common.parse_ax_fig
    common.get_date_format
"""

from .TimeSeriesDisplay import TimeSeriesDisplay
from .ContourDisplay import ContourDisplay
from .WindRoseDisplay import WindRoseDisplay
from .SkewTDisplay import SkewTDisplay
from .XSectionDisplay import XSectionDisplay
from .GeoDisplay import GeographicPlotDisplay
from .HistogramDisplay import HistogramDisplay
from .plot import Display
from . import common
from . import act_cmap
