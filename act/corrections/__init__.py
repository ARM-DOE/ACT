"""
=================================
act.corrections (act.corrections)
=================================

.. currentmodule:: act.corrections

The procedures in this module contain corrections for various datasets.

.. autosummary::
    :toctree: generated/

    ceil.correct_ceil
    doppler_lidar.correct_dl
    mpl.correct_mpl
    raman_lidar.correct_rl
    ship.correct_wind
"""

from . import ceil
from . import mpl
from . import ship
from . import doppler_lidar
from . import raman_lidar
