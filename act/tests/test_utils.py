import act
import xarray as xr
import numpy as np
from datetime import datetime


def test_dates_between():
    start_date = '20190101'
    end_date = '20190110'

    date_list = act.utils.dates_between(start_date, end_date)
    answer = [datetime(2019, 1, 1),
              datetime(2019, 1, 2),
              datetime(2019, 1, 3),
              datetime(2019, 1, 4),
              datetime(2019, 1, 5),
              datetime(2019, 1, 6),
              datetime(2019, 1, 7),
              datetime(2019, 1, 8),
              datetime(2019, 1, 9),
              datetime(2019, 1, 10)]
    assert date_list == answer


def add_in_nan():
    # Make a 1D array with a 4 day gap in the data
    time_list = [np.datetime64(datetime(2019, 1, 1)),
                 np.datetime64(datetime(2019, 1, 2)),
                 np.datetime64(datetime(2019, 1, 3)),
                 np.datetime64(datetime(2019, 1, 4)),
                 np.datetime64(datetime(2019, 1, 9))]
    data = np.linspace(0, 8, 5)

    time_list = xr.DataArray(time_list)
    data = xr.DataArray(data)
    data_filled, time_filled = act.utils.add_in_nan(
        time_list, data)
    assert(data_filled.data == np.array([0, 2, 4, 6,
                                         np.nan, np.nan, np.nan, np.nan,
                                         8]))
    time_answer = [datetime(2019, 1, 1),
                   datetime(2019, 1, 2),
                   datetime(2019, 1, 3),
                   datetime(2019, 1, 4),
                   datetime(2019, 1, 5),
                   datetime(2019, 1, 6),
                   datetime(2019, 1, 7),
                   datetime(2019, 1, 8),
                   datetime(2019, 1, 9)]
    assert(time_list == time_answer)
