import act


def test_download_surfrad():
    results = act.discovery.download_surfrad_data(
        site='tbl', startdate='20230601', enddate='20230602'
    )
    assert len(results) == 2
    assert 'tbl23152.dat' in results[0]
