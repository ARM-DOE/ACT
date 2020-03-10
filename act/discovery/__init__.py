"""
=============================
act.discovery (act.discovery)
=============================

.. currentmodule:: act.discovery

This module contains procedures for exploring and downloading data on
ARM Data Discovery.

.. autosummary::
    :toctree: generated/

    download_data
    croptype
    get_asos
"""

from .get_armfiles import download_data
from .get_CropScape import croptype
from .get_asos import get_asos
