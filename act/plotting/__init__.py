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
    Display
    WindRoseDisplay
    TimeSeriesDisplay
"""

from .plot import TimeSeriesDisplay
from .plot import Display, WindRoseDisplay
from . import common
