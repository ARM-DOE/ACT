"""
NASA MPLNET
-----------

This example shows how to download data from
NASA's MicroPulsed Lidar Network

"""

import act

# Determine MPLNET site of interest
meta = act.discovery.get_mplnet_meta(sites="active", method="data", print_to_screen=True)

# Download MPLNET data for site of interest
output = act.discovery.download_mplnet_data(
    version=3, level=1, product="NRB", site="GSFC", year="2022", month="09", day="01"
)
