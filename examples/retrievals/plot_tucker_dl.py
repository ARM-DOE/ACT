"""
Example for using the Tucker method to retrieve the planetary boundary layer height from Doppler lidar data.
============================================================================================================
The Tucker method uses the lag-1 autocorrelation of the radial velocity to estimate the atmospheric (turbulent) velocity variance and the instrument noise velocity variance.
The planetary boundary layer height is then estimated as the height of the first range gate whose atmospheric variance exceeds a specified threshold.
The example below uses a Doppler lidar dataset from the ARM SGP site, but the method can be applied to any Doppler lidar dataset with radial velocity data.

References:
Tucker, S. C., Hardesty, R. M., & Brewer, W. A. (2009). Estimating the planetary boundary layer height from Doppler lidar radial velocity measurements.
 Journal of Atmospheric and Oceanic Technology, 26(9), 1745-175
"""
import matplotlib.pyplot as plt

import act

dl_ds = act.io.read_arm_netcdf(act.tests.sample_files.EXAMPLE_DLFPT)
dl_ds = act.retrievals.calculate_tucker_method_pbl(dl_ds, interval="10min")

display = act.plotting.TimeSeriesDisplay(dl_ds, subplot_shape=(3, ), figsize=(12, 10))
display.plot('intensity', cmap='Spectral_r', vmin=1, vmax=1.5,
             subplot_index=(0, ))
display.plot('pbl_tucker', subplot_index=(0, ),
             label='PBL Height (m)', color='black', linestyle='-', linewidth=2)
display.plot('tucker_atmospheric_variance', subplot_index=(1, ), vmin=0, vmax=0.5,
             label='Atmospheric Variance', cmap='Spectral_r')
display.plot('tucker_noise_variance', subplot_index=(2, ), vmin=0, vmax=0.5,
             label='Noise Variance', cmap='Spectral_r')
display.set_yrng([0, 3000], subplot_index=(0, ))
display.set_yrng([0, 3000], subplot_index=(1, ))
display.set_yrng([0, 3000], subplot_index=(2, ))

plt.suptitle('Tucker Method PBL Height Retrieval', fontsize=16)
plt.show()
