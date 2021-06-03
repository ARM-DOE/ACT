"""
Module that containing utilities involving datetimes.

"""

import datetime as dt
import pandas as pd
import numpy as np
from scipy import stats
import warnings


def dates_between(sdate, edate):
    """
    Ths procedure returns all of the dates between *sdate* and *edate*.

    Parameters
    ----------
    sdate : str
        The string containing the start date. The string is formatted
        YYYYMMDD.
    edate : str
        The string containing the end date. The string is formatted
        YYYYMMDD.

    Returns
    -------
    all_dates : array of datetimes
        The array containing the dates between *sdate* and *edate*.

    """
    days = dt.datetime.strptime(edate, '%Y%m%d') - \
        dt.datetime.strptime(sdate, '%Y%m%d')
    all_dates = [dt.datetime.strptime(sdate, '%Y%m%d') + dt.timedelta(days=d)
                 for d in range(days.days + 1)]
    return all_dates


def numpy_to_arm_date(_date, returnTime=False):
    """
    Given a numpy datetime64, return an ARM standard date (yyyymmdd).

    Parameters
    ----------
    date : numpy.datetime64
        Numpy datetime64 date.
    returnTime : boolean
        If set to true, returns time instead of date

    Returns
    -------
    arm_date : string
        Returns an arm date.

    """
    date = pd.to_datetime(str(_date))
    if returnTime is False:
        date = date.strftime('%Y%m%d')
    else:
        date = date.strftime('%H%M%S')

    return date


def reduce_time_ranges(time, time_delta=60, broken_barh=False):
    """
    Given a time series, this function will return a list of tuples of time
    ranges representing the contineous times where no data is detected missing.

    Parameters
    ----------
    time : numpy datetime64 array
        The numpy array of date time values.
    time_delta : int
        The number of seconds to use as default time step in time array.
    broken_barh : boolean
        Option to return start time and duration instead of start time and
        end time. This is used with the pyplot.broken_barh() plotting routine.

    Returns
    -------
    time_ranges : list of tuples with 2 numpy datetime64 times
        The time range(s) of contineous data.

    """
    # Convert integer sections to numpy datetime64
    time_delta = np.timedelta64(int(time_delta * 1000), 'ms')

    # Make a difference array to find where time difference is great than time_delta
    diff = np.diff(time)
    dd = np.where(diff > time_delta)[0]

    if len(dd) == 0:
        return [(time[0], time[-1] - time[0])]

    # A add to start and end of array for beginning and end values
    dd = np.insert(dd, 0, -1)
    dd = np.append(dd, len(time) - 1)

    # Create a list of tuples containg time ranges or start time with duration
    if broken_barh:
        return [(time[dd[ii] + 1], time[dd[ii + 1]] - time[dd[ii] + 1])
                for ii in range(len(dd) - 1)]
    else:
        return [(time[dd[ii] + 1], time[dd[ii + 1]]) for ii in range(len(dd) - 1)]


def determine_time_delta(time, default=60):
    """
    Returns the most likely time step in seconds by analyzing the difference in
    time steps.

    Parameters
    ----------
    time : numpy datetime64 array
        The numpy array of date time values.
    default : int or float
        The default number to return if unable to calculate a value.

    Returns
    -------
    time_delta : float
        Returns the number of seconds for the most common time step. If can't
        calculate a value the default value is returned.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if time.size > 1:
            mode = stats.mode(np.diff(time))
            time_delta = mode.mode[0]
            time_delta = time_delta.astype('timedelta64[s]').astype(float)
        else:
            time_delta = default

    return float(time_delta)


def datetime64_to_datetime(time):
    """
    Given a numpy datetime64 array time series, return datetime (y, m, d, h, m, s)

    Parameters
    ----------
    time: numpy datetime64 array, list of numpy datetime64 values or scalar numpy datetime64
        The numpy array of date time values.

    Returns
    -------
    datetime: list
        Returns a list of datetimes (y, m, d, h, m, s) from a time series.

    """
    if isinstance(time, (tuple, list)):
        time = np.array(time)
    if len(time.shape) == 0:
        time = np.array([time])
    datetime_array = [dt.datetime.fromtimestamp(tm.astype('datetime64[ms]').astype('float') / 1000.,
                                                tz=dt.timezone.utc).replace(tzinfo=None) for tm in time]
    return datetime_array
