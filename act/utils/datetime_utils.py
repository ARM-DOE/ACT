import datetime as dt

def dates_between(sdate,edate):
    """
    Procedure dates_between
    -----------------------
    Ths procedure returns all of the dates between *sdate* and *edate.
    
    Parameters
    ----------
    sdate: datetime
        The datetime containing the start date.
    edate: datetime
        The datetime containing the end date.
    
    Returns
    -------
    all_dates: array of datetimes
        The array containing the dates between *sdate* and *edate*
    """
    
    days = dt.datetime.strptime(sdate,'%Y%m%d')-\
        dt.datetime.strptime(edate,'%Y%m%d')
 
    all_dates = [dt.datetime.strptime(edate,'%Y%m%d')+\
        dt.timedelta(days=d) for d in range(days.days+1)]

    return all_dates

