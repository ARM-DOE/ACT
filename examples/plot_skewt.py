"""
Skew-T plot of a sounding
-------------------------

This example shows how to make a Skew-T plot from a sounding
and calculate stability indicies.  METPy needs to be installed
in order to run this example

"""


import xarray as xr
from matplotlib import pyplot as plt

import act

# Make sure attributes are retained
xr.set_options(keep_attrs=True)

try:
    import metpy

    METPY = True
except ImportError:
    METPY = False

if METPY:
    # Read data
    sonde_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_SONDE1)

    print(list(sonde_ds))
    # Calculate stability indicies
    sonde_ds = act.retrievals.calculate_stability_indicies(
        sonde_ds, temp_name='tdry', td_name='dp', p_name='pres', rh_name='rh'
    )
    print(sonde_ds['lifted_index'])

    # Set up plot
    skewt = act.plotting.SkewTDisplay(sonde_ds, figsize=(15, 10))

    # Add data
    skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
    sonde_ds.close()
    plt.show()
