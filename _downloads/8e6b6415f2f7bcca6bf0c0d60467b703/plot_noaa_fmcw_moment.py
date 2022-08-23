"""
NOAA FMCW Moment Data
---------------------

ARM and NOAA have campaigns going on in the Crested Butte, CO region
and as part of that campaign NOAA has FMCW radars deployed that could
benefit the broader ARM and NOAA communities.  This example shows how
easy it is to download and read in NOAA PSL data.

"""

import act
import os
import matplotlib.pyplot as plt

# Use the ACT downloader to download a file from the
# Kettle Ponds site on 8/15/2022 at 2300 UTC
result = act.discovery.download_noaa_psl_data(
    site='kps', instrument='Radar FMCW Moment', startdate='20220815', hour='23'
)

# Read in the .raw file.  Spectra data are also downloaded
obj = act.io.noaapsl.read_psl_radar_fmcw_moment([result[-1]])

# As Part of the reading in, the script calculates ranges and
# corrects the SNR for range
display = act.plotting.TimeSeriesDisplay(obj)
display.plot('reflectivity_uncalibrated', cmap='jet', vmin=-20, vmax=40)
plt.show()
