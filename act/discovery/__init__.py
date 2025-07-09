"""
This module contains procedures for exploring and downloading data
from a variety of web services

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        'ameriflux',
        'arm',
        'asos',
        'airnow',
        'cropscape',
        'improve',
        'noaapsl',
        'neon',
        'surfrad',
        'nasa',
    ],
    submod_attrs={
        'ameriflux': ['download_ameriflux_data'],
        'arm': ['download_arm_data', 'download_arm_data_mod', 'get_arm_doi'],
        'asos': ['get_asos_data'],
        'airnow': ['get_airnow_bounded_obs', 'get_airnow_obs', 'get_airnow_forecast'],
        'cropscape': ['get_crop_type'],
        'improve': ['get_improve_data'],
        'noaapsl': ['download_noaa_psl_data'],
        'neon': ['get_neon_site_products', 'get_neon_product_avail', 'download_neon_data'],
        'surfrad': ['download_surfrad_data'],
        'nasa': ["download_mplnet_data", "get_mplnet_meta"],
    },
)
