import datetime as dt

def dates_between(sdate,edate):
    """Return all dates between 2 file dates"""
    days = dt.datetime.strptime(sdate,'%Y%m%d')-\
        dt.datetime.strptime(edate,'%Y%m%d')

    all_dates = [dt.datetime.strptime(edate,'%Y%m%d')+\
        dt.timedelta(days=d) for d in range(days.days+1)]

    return all_dates

