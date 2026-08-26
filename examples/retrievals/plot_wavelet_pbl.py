"""
Planetary Boundary Layer Height Wavelet Method Retrieval
---------------------------------------------------------

This example shows how to estimate the planetary boundary layer
height via a Haar wavelet covariance transform retrieval

Author: Robert Jackson
"""

import matplotlib.pyplot as plt
from arm_test_data import DATASETS

import act

# Read Ceilometer data for an example
filename_ceil = DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc')
ds = act.io.arm.read_arm_netcdf(filename_ceil)

# Apply corrections to the dataset
ds = act.corrections.correct_ceil(ds, var_name='backscatter')

# Estimate PBL Height via a Haar wavelet covariance transform,
# limiting the search to below 2000 m to exclude elevated cloud layers
ds = act.retrievals.pbl_lidar.calculate_wavelet_pbl(
    ds, var_name='backscatter', range_name='range', scale=60.0, max_height=1500.0
)

# Plot the pbl height estimates
display = act.plotting.TimeSeriesDisplay(ds, figsize=(10, 5))

# plot the CL backscatter before overlaying the Wavelet Method PBL Height
display.plot(
    'backscatter',
    cmap='ChaseSpectral',
    vmin=-6,
    vmax=6,
    set_title='SGP Ceilometer PBL Height Estimate via Wavelet Method',
)

# overlay the PBL Height estimate. We will compute a 10 minute running average to smooth the estimate.
# The rolling function is used to compute a running average of the PBL height estimate over a 10 minute window,
# with a minimum of 3 valid data points required for the average to be computed.
# The center=True argument ensures that the average is centered on the current time point.
display.axes[0].plot(
    ds['resampled_time'].values,
    ds['pbl_wavelet'].rolling(resampled_time=10, center=True, min_periods=3).mean().values,
    color='w',
    linewidth=2,
    label='Wavelet PBL Height Estimate',
)
# shorten the range
display.set_yrng([0, 2000])
plt.show()
