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
filename_mpl = DATASETS.fetch('sgpminimplC1.b1.20260330.090000.nc')
ds = act.io.arm.read_arm_netcdf(filename_mpl)

# range_bins ships in km; convert to meters so fit_min/max_height can be
# expressed in the meters for the PBLH scheme (and share units with 'height').
ds = ds.assign_coords(range_bins=ds['range_bins'].values * 1000.0)
ds['range_bins'].attrs['units'] = 'm'

# Apply corrections to the dataset
ds = act.corrections.correct_mpl(ds)

# Estimate PBL Height via a Haar wavelet covariance transform,
# limiting the search to below 2000 m to exclude elevated cloud layers
ds = act.retrievals.pbl_lidar.calculate_wavelet_pbl(
    ds, var_name="signal_return_cross_pol", range_name='range_bins', scale=60.0, max_height=1500.0
)

# Plot the pbl height estimates
display = act.plotting.TimeSeriesDisplay(ds, figsize=(10, 5))

# plot the CL backscatter before overlaying the Wavelet Method PBL Height
display.plot(
    "signal_return_cross_pol",
    cmap='HomeyerRainbow',
    vmin=-4,
    vmax=10,
    set_title='SGP miniMPL PBL Height Estimate via Wavelet Method',
)

# overlay the PBL Height estimate. We will compute a 10 minute running average to smooth the estimate.
# The rolling function is used to compute a running average of the PBL height estimate over a 10 minute window,
# with a minimum of 3 valid data points required for the average to be computed.
# The center=True argument ensures that the average is centered on the current time point.
display.axes[0].plot(
    ds['resampled_time'].values,
    ds['pbl_wavelet'].rolling(resampled_time=10, center=True, min_periods=3).mean().values,
    color='white',
    linewidth=2,
    label='Wavelet PBL Height Estimate',
)
# shorten the range
display.set_yrng([0, 2000])
plt.show()
