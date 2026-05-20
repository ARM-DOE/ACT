import sys

import numpy as np
import pytest

import act


@pytest.mark.skipif(
    not sys.platform.startswith('linux'),
    reason='To avoid exceeded downloads for testing, only test on linux',
)
def test_get_mplnet_meta():
    output = act.discovery.get_mplnet_meta(
        sites="GSFC", method="data", year="2024", month="09", day="12"
    )

    assert 'id' in output[0]
    assert 'station' in output[0]
    assert output[0]['station']['latitude_unit'] == "deg"

    with pytest.raises(ValueError):
        output = act.discovery.get_mplnet_meta()
    with pytest.raises(ValueError):
        output = act.discovery.get_mplnet_meta(sites=10)


@pytest.mark.skipif(
    not sys.platform.startswith('linux'),
    reason='To avoid exceeded downloads for testing, only test on linux',
)
def test_download_mplnet_data():
    output = act.discovery.download_mplnet_data(
        version=3, level=1, product="NRB", site="GSFC", year="2020", month="09", day="01"
    )

    assert len(output) == 1
    assert output[0][-3:] == "nc4"

    with pytest.raises(ValueError):
        output = act.discovery.download_mplnet_data()
    with pytest.raises(ValueError):
        output = act.discovery.download_mplnet_data(version=3)
    with pytest.raises(ValueError):
        output = act.discovery.download_mplnet_data(version=3, level=1)
    with pytest.raises(ValueError):
        output = act.discovery.download_mplnet_data(version=3, level=1, product='NRB')
    with pytest.raises(ValueError):
        output = act.discovery.download_mplnet_data(version=3, level=1, product='NRB', site="GSFC")
    with pytest.raises(ValueError):
        output = act.discovery.download_mplnet_data(
            version=3, level=1, product='NRB', site="GSFC", year="2020"
        )
