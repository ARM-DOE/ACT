"""
The procedures in this module contain corrections for various datasets.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['ceil', 'doppler_lidar', 'mpl', 'raman_lidar', 'ship'],
    submod_attrs={
        'ceil': ['*'],
        'doppler_lider': ['*'],
        'mpl': ['*'],
        'raman_lidar': ['*'],
        'ship': ['*'],
    },
)
