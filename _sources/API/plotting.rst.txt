.. automodule:: act.plotting
   :members:

act.plotting contains classes for displaying data. :func:`act.plotting.Display` is the
base class on which all other Display classes are inherited from. If you are making
a new Display object, please make it inherited from this class. 

:func:`act.plotting.TimeSeriesDisplay` handles the plotting of timeseries. 
:func:`act.plotting.WindRoseDisplay` handles the plotting of wind rose plots.
:func:`act.plotting.SkewTDisplay` handles the plotting of Skew-T diagrams.
:func:`act.plotting.XSectionDisplay` handles the plotting of cross sections.

.. toctree::
   :maxdepth: 2

   Display
   TimeSeriesDisplay
   ContourDisplay
   WindRoseDisplay
   SkewTDisplay
   XSectionDisplay
   HistogramDisplay

