import numpy as np

import act


def test_noaa_psl():
    result = act.discovery.download_noaa_psl_data(
        site='ctd',
        instrument='Parsivel',
        startdate='20211231',
        enddate='20220101',
        output='./data/',
    )
    assert len(result) == 48

    result = act.discovery.download_noaa_psl_data(
        site='ctd', instrument='Pressure', startdate='20220101', hour='00'
    )
    assert len(result) == 1

    result = act.discovery.download_noaa_psl_data(
        site='ctd', instrument='GpsTrimble', startdate='20220104', hour='00'
    )
    assert len(result) == 6

    types = [
        'Radar S-band Moment',
        'Radar S-band Bright Band',
        '449RWP Bright Band',
        '449RWP Wind',
        '449RWP Sub-Hour Wind',
        '449RWP Sub-Hour Temp',
        '915RWP Wind',
        '915RWP Temp',
        '915RWP Sub-Hour Wind',
        '915RWP Sub-Hour Temp',
    ]
    for t in types:
        result = act.discovery.download_noaa_psl_data(
            site='ctd', instrument=t, startdate='20220601', hour='01'
        )
        assert len(result) == 1

    types = ['Radar FMCW Moment', 'Radar FMCW Bright Band']
    files = [3, 1]
    for i, t in enumerate(types):
        result = act.discovery.download_noaa_psl_data(
            site='bck', instrument=t, startdate='20220101', hour='01'
        )
        assert len(result) == files[i]

    with np.testing.assert_raises(ValueError):
        result = act.discovery.download_noaa_psl_data(
            instrument='Parsivel', startdate='20220601', hour='01'
        )
    with np.testing.assert_raises(ValueError):
        result = act.discovery.download_noaa_psl_data(
            site='ctd', instrument='dongle', startdate='20220601', hour='01'
        )
