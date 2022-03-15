"""
Example on how to retrieve stability indicies from a sounding
-------------------------------------------------------------

This example shows how to retrieve CAPE, CIN, and lifted index
from a sounding.
"""
import act

sonde_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_SONDE1)

sonde_ds = act.retrievals.calculate_stability_indicies(
    sonde_ds, temp_name='tdry', td_name='dp', p_name='pres'
)
print(
    'Lifted index = '
    + str(sonde_ds['lifted_index'].values)
    + ' '
    + str(sonde_ds['lifted_index'].units)
)
print(
    'Surface based CAPE = '
    + str(sonde_ds['surface_based_cape'].values)
    + ' '
    + str(sonde_ds['surface_based_cape'].units)
)
print(
    'Surface based CIN = '
    + str(sonde_ds['surface_based_cin'].values)
    + ' '
    + str(sonde_ds['surface_based_cin'].units)
)
print(
    'Most unstable CAPE = '
    + str(sonde_ds['most_unstable_cape'].values)
    + ' '
    + str(sonde_ds['most_unstable_cape'].units)
)
print(
    'Most unstable CIN = '
    + str(sonde_ds['most_unstable_cin'].values)
    + ' '
    + str(sonde_ds['most_unstable_cin'].units)
)
print(
    'LCL temperature = '
    + str(sonde_ds['lifted_condensation_level_temperature'].values)
    + ' '
    + str(sonde_ds['lifted_condensation_level_temperature'].units)
)
print(
    'LCL pressure = '
    + str(sonde_ds['lifted_condensation_level_pressure'].values)
    + ' '
    + str(sonde_ds['lifted_condensation_level_pressure'].units)
)
sonde_ds.close()
