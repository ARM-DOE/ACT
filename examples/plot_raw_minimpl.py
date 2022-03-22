"""
Read and plot a PPI from raw mini-MPL data
------------------------------------------

Example of how to read in raw data from the mini-MPL
and plot out the PPI by converting it to PyART

Author: Adam Theisen
"""


from matplotlib import pyplot as plt

import act

try:
    import pyart

    PYART_AVAILABLE = True
except ImportError:
    PYART_AVAILABLE = False

# Read in sample mini-MPL data
files = act.tests.sample_files.EXAMPLE_SIGMA_MPLV5
obj = act.io.mpl.read_sigma_mplv5(files)

# Create a PyART Radar Object
radar = act.utils.create_pyart_obj(
    obj, azimuth='azimuth_angle', elevation='elevation_angle', range_var='range'
)

# Creat Plot Display
if PYART_AVAILABLE:
    display = pyart.graph.RadarDisplay(radar)
    display.plot('nrb_copol', sweep=0, title_flag=False, vmin=0, vmax=1.0, cmap='jet')
    plt.show()
