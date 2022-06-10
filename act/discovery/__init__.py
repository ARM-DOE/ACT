"""
This module contains procedures for exploring and downloading data
from a variety of web services

"""

from .get_armfiles import download_data
from .get_asos import get_asos
from .get_cropscape import croptype
from .get_airnow import get_airnow_bounded_obs, get_airnow_obs, get_airnow_forecast

