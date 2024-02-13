"""
This module contains procedures for reading and writing various ARM datasets.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['arm', 'text', 'icartt', 'mpl', 'neon', 'noaagml', 'noaapsl', 'pysp2', 'hysplit'],
    submod_attrs={
        'arm': [
            'WriteDataset',
            'check_arm_standards',
            'create_ds_from_arm_dod',
            'read_arm_netcdf',
            'check_if_tar_gz_file',
            'read_arm_mmcr',
        ],
        'text': ['read_csv'],
        'icartt': ['read_icartt'],
        'mpl': ['proc_sigma_mplv5_read', 'read_sigma_mplv5'],
        'neon': ['read_neon_csv'],
        'noaagml': [
            'read_gml',
            'read_gml_co2',
            'read_gml_halo',
            'read_gml_met',
            'read_gml_ozone',
            'read_gml_radiation',
            'read_surfrad',
        ],
        'noaapsl': [
            'read_psl_wind_profiler',
            'read_psl_wind_profiler_temperature',
            'read_psl_parsivel',
            'read_psl_radar_fmcw_moment',
            'read_psl_surface_met',
        ],
        'pysp2': ['read_hk_file', 'read_sp2', 'read_sp2_dat'],
        'sodar': ['read_mfas_sodar'],
        'hysplit': ['read_hysplit'],
    },
)
