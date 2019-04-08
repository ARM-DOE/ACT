import act
import numpy as np

from matplotlib import pyplot as plt

sonde_ds = act.io.armfiles.read_netcdf(
    act.tests.sample_files.EXAMPLE_MET_WILDCARD)

WindDisplay = act.plotting.WindRoseDisplay(sonde_ds)
WindDisplay.plot('wdir_vec_mean', 'wspd_vec_mean',
                 spd_bins=np.linspace(0, 15, 5), num_dirs=30,
                 tick_interval=1)
plt.show()
