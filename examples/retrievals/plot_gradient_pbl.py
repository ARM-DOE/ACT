"""
Planetary Boundary Layer Height Gradient Method Retrievals
----------------------------------------------------------

This example shows how to estimate the planetary boundary layer
height via a gradient method retrieval during the REAL-SGP activity.

Author: Joe O'Brien
"""

from arm_test_data import DATASETS

import act

# Read Ceilometer data for an example
filename_mpl = DATASETS.fetch('sgpminimplC1.b1.20260330.090000.nc')
ds = act.io.arm.read_arm_netcdf(filename_mpl)

# range_bins ships in km; convert to meters so min/max_height can be
# expressed in the meters for the pbl scheme.
ds = ds.assign_coords(range_bins=ds['range_bins'].values * 1000.0)
ds['range_bins'].attrs['units'] = 'm'

# Apply corrections to the dataset
ds = act.corrections.correct_mpl(ds)

# Estimate PBL Height via a gradient method. max_height keeps the search for
# the sharpest negative gradient within a physically plausible PBL range --
# beyond that, backscatter is noise-dominated and an unbounded search can
# lock onto that noise instead of the real aerosol-layer top.
ds = act.retrievals.pbl_lidar.calculate_gradient_pbl(
    ds,
    parm="signal_return_cross_pol",
    dis_parm="range_bins",
    smooth_dis=3,
    min_height=100.0,
    max_height=2500.0,
)

# Estimate PBL Height via a modified gradient method
ds = act.retrievals.pbl_lidar.calculate_modified_gradient_pbl(
    ds,
    parm="signal_return_cross_pol",
    threshold=1e-3,
    smooth_dis=3,
    max_height=2500.0,
    min_height=100.0,
    dis_parm="range_bins",
)

# Plot the pbl height estimates
display = act.plotting.TimeSeriesDisplay(ds, subplot_shape=(2,), figsize=(10, 8))

# plot the CL backscatter before overlaying the Gradient Method PBL Height
display.plot(
    "signal_return_cross_pol",
    subplot_index=(0,),
    cmap='HomeyerRainbow',
    vmin=-4,
    vmax=10,
    set_title='SGP MiniMPL with PBL Height Estimate via Gradient Method',
)

# overlay the PBL Height estimate, compute ~10min temporal averages
display.axes[0].plot(
    ds['time'].values,
    ds['pbl_gradient'].rolling(time=38, min_periods=3, center=True).mean().values,
    color='white',
)
# shorten the range
display.set_yrng([0, 3000], subplot_index=(0,))

# plot the CL backscatter before overlaying the Modified Gradient PBL Height
display.plot(
    "signal_return_cross_pol",
    subplot_index=(1,),
    cmap='HomeyerRainbow',
    vmin=-4,
    vmax=10,
    set_title='SGP MiniMPL with PBL Height Estimate via Modified Gradient Method',
)

# overlay the PBL Height estimate, compute ~10min temporal averages
display.axes[1].plot(
    ds['time'].values,
    ds['pbl_mod_gradient'].rolling(time=38, min_periods=3, center=True).mean().values,
    color='white',
)
# shorten the range
display.set_yrng([0, 3000], subplot_index=(1,))
