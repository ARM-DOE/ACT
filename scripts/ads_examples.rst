=======================
ARM Data Surveyor (ADS)
=======================

Given the shear number of functions supported in ADS, these examples are not exhaustive.  They
are meant to search as a starting point.  A full list of options can be found by executing::

    python ads.py -h

EXAMPLES:
~~~~~~~~~

Plot a simple timeseries plot::

    python ads.py -f ../act/tests/data/sgpmetE13.b1.20190101.000000.cdf -o ./image.png -fd temp_mean -ts --plot

Plot a simple timeseries plot with the QC block plot in a second plot::

    python ads.py -f ../act/tests/data/sgpmetE13.b1.20190101.000000.cdf -o ./image.png -fds temp_mean temp_mean  -pt plot qc -mp -ts -si "(0,), (1,)" -ss 2
