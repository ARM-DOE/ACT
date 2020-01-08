===============
Release History
===============

0.1.6 (Released 2019-12-02)
---------------------------

Numerous updates have been made since the last release.

* The ACT object has been updated to remove the "act" attributes
  and instead store that information in xarrays attrs.
  Code has also been updated to not need additional information
  that is adding in during the reading so that pure xarray datasets can be used.

* Corrections to the doppler lidar correction, wind barb plotting, sonde stability calculations.

* Adding doppler lidar wind retrievals, code to calculate destination lat/lon
  from starting lat/lon and heading/distance


0.1.6 (Released 2019-10-10)
---------------------------

* Corrections for Doppler Lidar, Raman Lidar added as well as updates to MPL correction code
* New plotting capability for plotting size spectra
* Added capability to calculate course and speed over ground from lat/lon data
* Added capability to correct wind speed and direction for ship motion
* Removing ACT rolling_window function

0.1.5 (Released 2019-09-20)
---------------------------

The recent update to xarray caused some issues with the tests.
These have been updated to accommodate

0.1.4 (Released 2019-09-20)
---------------------------

Version includes new plotting functions for visualizing
bit-packed QC as used by the ARM program. A new function
for calculating precipitable water vapor from a sounding
was also created and including. Other updates include bug
fixes and the inclusion of a roadmap.

0.1.3 (Released 2019-09-06)
---------------------------

Merge pull request #116 from AdamTheisen/master

Missing requirements and data files for tests

0.1.2 (Released 2019-09-06)
---------------------------

Some .txt files were missing from the pip installs.

0.1.1 (Released 2019-09-03)
---------------------------

This release is to support the anaconda distribution.

0.1 (Released 2019-09-03)
-------------------------

Additional documentation for the QC functions have been added.
New functions for weighted averaging of time-series data, precipitation
accumulation, and cloud base height detection using a sobel edge
detection method have also been added.

0.0.2 (Released 2019-08-14)
---------------------------

Secondary release to work with pypi

0.0.1 (Released 2019-08-14)
---------------------------

This is the initial release of the ACT toolkit for use in working with
atmospheric time-series data. Library includes scripts for accessing
data through APIs, I/O scripts, data visualization, quality control
algorithms, corrections, and retrievals.