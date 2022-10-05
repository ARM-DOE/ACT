"""
This module contains procedures for reading and writing various ARM datasets.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['armfiles', 'csvfiles', 'mpl', 'noaagml', 'noaapsl', 'pysp2'],
    submod_attrs={
        'armfiles': [
            'WriteDataset',
            'check_arm_standards',
            'create_obj_from_arm_dod',
            'read_netcdf',
        ],
        'csvfiles': 'read_csv',
        'mpl': ['proc_sigma_mplv5_read', 'read_sigma_mplv5'],
        'read_gml': [
            'read_gml',
            'read_gml_co2',
            'read_gml_halo',
            'read_gml_met',
            'read_gml_ozone',
            'read_gml_radiation',
        ],
        'noaapsl': 'read_psl_wind_profiler',
        'pysp2': ['read_hk_file', 'read_sp2', 'read_sp2_dat'],
    },
)
