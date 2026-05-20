import pytest

import act


def test_download_surfrad():
    results = act.discovery.download_surfrad_data(
        site='tbl', startdate='20230601', enddate='20230602'
    )
    assert len(results) == 2
    assert 'tbl23152.dat' in results[0]

    results = act.discovery.download_surfrad_data(site='tbl', startdate='20230601', enddate=None)
    assert len(results) == 1
    assert 'tbl23152.dat' in results[0]

    pytest.raises(ValueError, act.discovery.download_surfrad_data, site=None, startdate=None)
