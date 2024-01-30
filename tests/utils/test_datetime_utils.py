from datetime import datetime

import numpy as np
import pandas as pd

import act


def test_dates_between():
    start_date = '20191201'
    end_date = '20201201'
    date_list = act.utils.dates_between(start_date, end_date)
    start_string = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
    end_string = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d')
    answer = np.arange(start_string, end_string, dtype='datetime64[D]')
    answer = np.append(answer, answer[-1] + 1)
    answer = answer.astype('datetime64[s]').astype(int)
    answer = [datetime.utcfromtimestamp(ii) for ii in answer]

    assert date_list == answer


def test_datetime64_to_datetime():
    time_datetime = [
        datetime(2019, 1, 1, 1, 0),
        datetime(2019, 1, 1, 1, 1),
        datetime(2019, 1, 1, 1, 2),
        datetime(2019, 1, 1, 1, 3),
        datetime(2019, 1, 1, 1, 4),
    ]

    time_datetime64 = [
        np.datetime64(datetime(2019, 1, 1, 1, 0)),
        np.datetime64(datetime(2019, 1, 1, 1, 1)),
        np.datetime64(datetime(2019, 1, 1, 1, 2)),
        np.datetime64(datetime(2019, 1, 1, 1, 3)),
        np.datetime64(datetime(2019, 1, 1, 1, 4)),
    ]

    time_datetime64_to_datetime = act.utils.datetime_utils.datetime64_to_datetime(time_datetime64)
    assert time_datetime == time_datetime64_to_datetime


def test_reduce_time_ranges():
    time = pd.date_range(start='2020-01-01T00:00:00', freq='1min', periods=100)
    time = time.to_list()
    time = time[0:50] + time[60:]
    result = act.utils.datetime_utils.reduce_time_ranges(time)
    assert len(result) == 2
    assert result[1][1].minute == 39

    result = act.utils.datetime_utils.reduce_time_ranges(time, broken_barh=True)
    assert len(result) == 2


def test_date_parser():
    datestring = '20111001'
    output_format = '%Y/%m/%d'

    test_string = act.utils.date_parser(datestring, output_format, return_datetime=False)
    assert test_string == '2011/10/01'

    test_datetime = act.utils.date_parser(datestring, output_format, return_datetime=True)
    assert test_datetime == datetime(2011, 10, 1)


def test_date_parser_minute_second():
    date_string = '2020-01-01T12:00:00'
    parsed_date = act.utils.date_parser(date_string, return_datetime=True)
    assert parsed_date == datetime(2020, 1, 1, 12, 0, 0)

    output_format = parsed_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    assert output_format == '2020-01-01T12:00:00.000Z'


def test_adjust_timestamp():
    file = act.tests.sample_files.EXAMPLE_EBBR1
    ds = act.io.arm.read_arm_netcdf(file)
    ds = act.utils.datetime_utils.adjust_timestamp(ds)
    assert ds['time'].values[0] == np.datetime64('2019-11-24T23:30:00.000000000')

    ds = act.utils.datetime_utils.adjust_timestamp(ds, offset=-60 * 60)
    assert ds['time'].values[0] == np.datetime64('2019-11-24T22:30:00.000000000')
