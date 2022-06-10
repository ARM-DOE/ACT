"""
This module contains the common procedures used by all modules of the ARM
Community Toolkit.

"""

from .data_utils import (
    ChangeUnits,
    accumulate_precip,
    add_in_nan,
    assign_coordinates,
    convert_units,
    create_pyart_obj,
    get_missing_value,
    ts_weighted_average,
    height_adjusted_pressure,
    height_adjusted_temperature,
    convert_to_potential_temp
)
from .datetime_utils import (
    dates_between,
    datetime64_to_datetime,
    determine_time_delta,
    numpy_to_arm_date,
    reduce_time_ranges,
    date_parser,
)
from .geo_utils import add_solar_variable, destination_azimuth_distance
from .inst_utils import decode_present_weather
from .qc_utils import calculate_dqr_times
from .radiance_utils import planck_converter
from .ship_utils import calc_cog_sog
