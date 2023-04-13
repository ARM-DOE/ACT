"""
Retrieve stability indicies from a sounding
-------------------------------------------------------------

This example shows how to retrieve CAPE, CIN, and lifted index
from a sounding.
"""

import warnings

import act

warnings.filterwarnings('ignore')


def print_summary(ds, variables):
    for var_name in variables:
        print(f'{var_name}: {ds[var_name].values} ' f"units={ds[var_name].attrs['units']}")
    print()


sonde_ds = act.io.armfiles.read_netcdf(act.tests.sample_files.EXAMPLE_SONDE1)

sonde_ds = act.retrievals.calculate_stability_indicies(
    sonde_ds, temp_name='tdry', td_name='dp', p_name='pres'
)

variables = [
    'lifted_index',
    'surface_based_cape',
    'surface_based_cin',
    'most_unstable_cape',
    'most_unstable_cin',
    'lifted_condensation_level_temperature',
    'lifted_condensation_level_pressure',
]

print_summary(sonde_ds, variables)
