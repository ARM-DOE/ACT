"""
This module contains classes for displaying data.
:func:`act.plotting.Display` is the
base class on which all other Display classes are inherited from. If you are making
a new Display object, please make it inherited from this class.

| :func:`act.plotting.ContourDisplay` handles the plotting of contour plots.
| :func:`act.plotting.DistributionDisplay` handles the plotting of distribution-related plots.
| :func:`act.plotting.GeographicPlotDisplay` handles the plotting of lat-lon plots.
| :func:`act.plotting.SkewTDisplay` handles the plotting of Skew-T diagrams.
| :func:`act.plotting.TimeSeriesDisplay` handles the plotting of timeseries.
| :func:`act.plotting.WindRoseDisplay` handles the plotting of wind rose plots.
| :func:`act.plotting.XSectionDisplay` handles the plotting of cross sections.

"""

import lazy_loader as lazy

# Load colormaps
import cmweather  # noqa

# Eagerly load in common
from . import common  # noqa

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        'act_cmap',
        '_act_cmap',
        'common',
        'contourdisplay',
        'geodisplay',
        'plot',
        'skewtdisplay',
        'timeseriesdisplay',
        'windrosedisplay',
        'xsectiondisplay',
        'distributiondisplay',
    ],
    submod_attrs={
        'contourdisplay': ['ContourDisplay'],
        'geodisplay': ['GeographicPlotDisplay'],
        'plot': ['Display', 'GroupByDisplay'],
        'skewtdisplay': ['SkewTDisplay'],
        'timeseriesdisplay': ['TimeSeriesDisplay'],
        'windrosedisplay': ['WindRoseDisplay'],
        'xsectiondisplay': ['XSectionDisplay'],
        'distributiondisplay': ['DistributionDisplay'],
    },
)
