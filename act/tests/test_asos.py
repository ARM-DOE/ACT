import numpy as np
from act.discovery import get_asos
from datetime import datetime


def test_get_ord():
    time_window = [datetime(2020, 2, 4, 2, 0), datetime(2020, 2, 12, 10, 0)]
    my_asoses = get_asos(time_window, station="ORD")
    assert "ORD" in my_asoses.keys()
    assert np.all(
        np.equal(my_asoses["ORD"]["sknt"].values[:10],
                 np.array([13., 11., 11., 11., 9., 10., 10., 11., 11., 11.])))


def test_get_region():
    my_keys = ['MDW', 'IGQ', 'ORD', '06C', 'PWK', 'LOT', 'GYY']
    time_window = [datetime(2020, 2, 4, 2, 0), datetime(2020, 2, 12, 10, 0)]
    lat_window = (41.8781 - 0.5, 41.8781 + 0.5)
    lon_window = (-87.6298 - 0.5, -87.6298 + 0.5)
    my_asoses = get_asos(time_window, lat_range=lat_window, lon_range=lon_window)
    asos_keys = [x for x in my_asoses.keys()]
    assert asos_keys == my_keys
