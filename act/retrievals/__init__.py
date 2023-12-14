"""
This module contains various retrievals for datasets.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['aeri', 'cbh', 'doppler_lidar', 'irt', 'radiation', 'sonde', 'sp2'],
    submod_attrs={
        'aeri': ['aeri2irt'],
        'cbh': ['generic_sobel_cbh'],
        'doppler_lidar': ['compute_winds_from_ppi'],
        'irt': ['sst_from_irt', 'sum_function_irt'],
        'radiation': [
            'calculate_dsh_from_dsdh_sdn',
            'calculate_irradiance_stats',
            'calculate_longwave_radiation',
            'calculate_net_radiation',
        ],
        'sonde': [
            'calculate_pbl_liu_liang',
            'calculate_precipitable_water',
            'calculate_stability_indicies',
            'calculate_pbl_heffter',
        ],
        'sp2': ['calc_sp2_diams_masses', 'process_sp2_psds'],
    },
)
