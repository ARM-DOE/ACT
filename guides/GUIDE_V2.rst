===========================
ACT Version 2 Release Guide
===========================

In preparation for version 2.0.0 of ACT, codes were standardized for consistency purposes as further defined in the `Contributor's Guide <https://arm-doe.github.io/ACT/userguide/CONTRIBUTING.html>`_.  These changes will break some users code as the API has changed.  This guide will detail the changes for each module.

Discovery
=========
Functionality has not changed but the naming of the API have changed for all discovery scripts to be more consistent and streamlined in their naming.

+------------------------------+------------------------------+
|Existing Function             | New Function                 |
+==============================+==============================+
| get_armfiles.download_data   | arm.download_arm_data        |
+------------------------------+------------------------------+
| get_armfiles.get_arm_doi     | arm.get_arm_doi              |
+------------------------------+------------------------------+
| get_asos.get_asos            | asos.get_asos_data           |
+------------------------------+------------------------------+
| get_airnow.*                 | airnow.*   Func Names Same   |
+------------------------------+------------------------------+
| get_cropscape.croptype       | cropscape.get_crop_type      |
+------------------------------+------------------------------+
| get_noaapsl.                 | noaapsl.                     |
|     download_noaa_psl_data   |     download_noaa_psl_data   |
+------------------------------+------------------------------+
| get_neon.get_site_products   | neon.get_neon_site_products  |
+------------------------------+------------------------------+
| get_neon.get_product_avail   | neon.get_neon_product_avail  |
+------------------------------+------------------------------+
| get_neon.download_neon_data  | neon.download_neon_data      |
+------------------------------+------------------------------+
| get_surfrad.download_surfrad | surfrad.download_surfrad_data|
+------------------------------+------------------------------+

IO
==
Similar to the discovery module, functionality has not changed but the naming convention has for similar reasoning.

+------------------------------+------------------------------+
|Existing Function             | New Function                 |
+==============================+==============================+
| armfiles                     | act.io.arm                   |
+------------------------------+------------------------------+
| armfiles.read_netcdf()       | arm.read_arm_netcdf          |
+------------------------------+------------------------------+
| armfiles.read_mmcr           | arm.read_arm_mmcr            |
+------------------------------+------------------------------+
| csvfiles                     | csv                          |
+------------------------------+------------------------------+

Plotting
========
A major change to how secondary y-axes are handled was implemented in the TimeSeriesDisplay and DistributionDisplay modules.  Currently, those plotting routines return a 1-D array of display axes.  This has always made the secondary y-axis more difficult to configure and use.  In the new version, it will return a 2-D array of display axes [[left axes, right axes]] to make it simpler to utilize.

HistogramDisplay is being renamed to DistributionDisplay to be more inclusive of the variety of visualization types that are housed there.  Additionally there are changes to two of the plot names to be more consistent with the others.

+------------------------------+------------------------------+
|Existing Function             | New Function                 |
+==============================+==============================+
| HistogramDisplay.            | DistributionDisplay.         |
|     plot_stacked_bar_graph   |     plot_stacked_bar         |
+------------------------------+------------------------------+
| HistogramDisplay.            | DistributionDisplay.         |
|     plot_stairstep_graph     |     plot_stairstep           |
+------------------------------+------------------------------+

Stamen maps for the GeoographicPlotDisplay are being retired.  Those maps will no longer be availabe at the end of October 2023.  The function was updated so that users can pass an image tile in.

QC
==
* The default behaviour for act.qc.qcfilter.datafilter is changing so that del_qc_var=False.  Previously, the default was to delete the QC variable after applying the QC.  Going forward it will not default to deleting the QC variables.

* ARM DQR webservice is being upgraded and the corresponding function will be upgraded to utilize this new webservice.

Tests
=====
Test data that have been historically stored in the act/tests/data area will be moved to a separate repository in order to reduce the package install size.
