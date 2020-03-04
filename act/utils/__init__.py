"""
=====================
act.utils (act.utils)
=====================

.. currentmodule:: act.utils

This module contains the common procedures used by all modules of the ARM
Community Toolkit.

.. autosummary::
    :toctree: generated/

    accumulate_precip
    add_in_nan
    assign_coordinates
    calc_cog_sog
    calculate_dqr_times
    convert_units
    dates_between
    datetime64_to_datetime
    decode_present_weather
    destination_azimuth_distance
    determine_time_delta
    get_missing_value
    numpy_to_arm_date
    planck_converter
    reduce_time_ranges
    ts_weighted_average
"""

from .data_utils import add_in_nan
from .data_utils import get_missing_value
from .data_utils import convert_units
from .data_utils import assign_coordinates
from .data_utils import accumulate_precip
from .data_utils import ts_weighted_average
from .datetime_utils import dates_between
from .datetime_utils import numpy_to_arm_date
from .datetime_utils import reduce_time_ranges
from .datetime_utils import determine_time_delta
from .datetime_utils import datetime64_to_datetime
from .qc_utils import calculate_dqr_times
from .ship_utils import calc_cog_sog
from .geo_utils import destination_azimuth_distance
from .inst_utils import decode_present_weather
from .radiance_utils import planck_converter
