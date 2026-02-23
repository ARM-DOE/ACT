"""
Retrieve stability indicies from a sounding
-------------------------------------------------------------

This example shows how to retrieve CAPE, CIN, and lifted index
from a sounding.
"""

import warnings

from arm_test_data import DATASETS

import act

warnings.filterwarnings('ignore')


def print_summary(ds, variables):
    for var_name in variables:
        print(f'{var_name}: {ds[var_name].values} ' f"units={ds[var_name].attrs['units']}")
    print()


filename_sonde = DATASETS.fetch('sgpsondewnpnC1.b1.20190101.053200.cdf')
sonde_ds = act.io.arm.read_arm_netcdf(filename_sonde)

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
