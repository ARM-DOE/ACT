"""
Planetary Boundary Layer Height Profile Fit Retrievals
----------------------------------------------------------

This example shows how to estimate the planetary boundary layer
height via a Profile Method scheme,
where a backscatter profile is fit to an idealized profile
via an error function using non-linear least-squares optimization.

This example uses the miniMPL data from the REAL-SGP activity.

Author: Joe O'Brien
"""

from arm_test_data import DATASETS

import act

# Read Ceilometer data for an example
filename_mpl = DATASETS.fetch('sgpminimplC1.b1.20260330.090000.nc')
ds = act.io.arm.read_arm_netcdf(filename_mpl)

# range_bins ships in km; convert to meters so fit_min/max_height can be
# expressed in the meters for the PBLH scheme (and share units with 'height').
ds = ds.assign_coords(range_bins=ds['range_bins'].values * 1000.0)
ds['range_bins'].attrs['units'] = 'm'

# Estimate PBL Height via a Profile Method
ds = act.retrievals.pbl_lidar.calculate_profile_fit_pbl(
    ds,
    parm="signal_return_cross_pol",
    dis_parm="range_bins",
    fit_min_height=100.0,
    fit_max_height=3500.0,
)

# Apply the ceilometer correction to the backscatter variable for plotting
# Note - after the PBL Height retrieval.
ds = act.corrections.correct_mpl(ds)

# Plot the pbl height estimates
display = act.plotting.TimeSeriesDisplay(ds, subplot_shape=(1,), figsize=(10, 8))

# plot the CL backscatter before overlaying the Gradient Method PBL Height
display.plot(
    'signal_return_cross_pol',
    subplot_index=(0,),
    cmap='HomeyerRainbow',
    vmin=-4,
    vmax=10,
    set_title='SGP miniMPL PBL Height Estimate via Profile Fit Method',
)

# overlay the PBL Height estimate, compute ~10min temporal averages
display.axes[0].plot(
    ds['time'].resample(time="30min").mean().values,
    ds['pbl_profile_fit'].resample(time="30min").mean().values,
    color='white',
)
# shorten the range
display.set_yrng([0, 3000], subplot_index=(0,))
