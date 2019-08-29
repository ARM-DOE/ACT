"""
act.utils.datetime_utils
------------------------

Module that containing utilities involving datetimes.

"""
import datetime as dt
import pandas as pd


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


def numpy_to_arm_date(_date):
    """
    Given a numpy datetime64, return an ARM standard date (yyyymmdd).

    Parameters
    ----------
    date : numpy.datetime64
        Numpy datetime64 date.

    Returns
    -------
    arm_date : string
        Returns an arm date.

    """
    date = pd.to_datetime(str(_date))
    date = date.strftime('%Y%m%d')

    return date
