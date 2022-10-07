"""
This module contains procedures for exploring and downloading data
from a variety of web services

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['get_armfiles', 'get_cropscape', 'get_cropscape', 'get_noaa_psl'],
    submod_attrs={
        'get_armfiles': ['download_data'],
        'get_asos': ['get_asos'],
        'get_airnow': ['get_airnow_bounded_obs', 'get_airnow_obs', 'get_airnow_forecast'],
        'get_cropscape': ['croptype'],
        'get_noaapsl': ['download_noaa_psl_data'],
    },
)
