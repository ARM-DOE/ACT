from datetime import datetime

import numpy as np
import pytest

import act


def test_get_ord():
    time_window = [datetime(2020, 2, 4, 2, 0), datetime(2020, 2, 12, 10, 0)]
    my_asoses = act.discovery.get_asos_data(time_window, station='ORD', regions='IL')
    assert 'ORD' in my_asoses.keys()
    assert np.all(
        np.equal(
            my_asoses['ORD']['sknt'].values[:10],
            np.array([13.0, 11.0, 14.0, 14.0, 13.0, 11.0, 14.0, 13.0, 13.0, 13.0]),
        )
    )


def test_get_latlon():
    expected = ['MDW', 'IGQ', 'ORD', '06C', 'PWK', 'LOT', 'GYY']

    time_window = [
        datetime(2020, 2, 10, 2, 0),
        datetime(2020, 2, 12, 10, 0),
    ]

    lat_window = (41.8781 - 0.5, 41.8781 + 0.5)
    lon_window = (-87.6298 - 0.5, -87.6298 + 0.5)

    my_asoses = act.discovery.get_asos_data(
        time_window,
        lat_range=lat_window,
        lon_range=lon_window,
    )

    assert sorted(my_asoses) == sorted(expected)


def test_get_region_by_name():
    time_window = [
        datetime(2020, 2, 4, 2, 0),
        datetime(2020, 2, 4, 4, 0),
    ]

    my_asoses = act.discovery.get_asos_data(
        time_window,
        regions="IL",
    )

    assert isinstance(my_asoses, dict)
    assert "ORD" in my_asoses
    assert "MDW" in my_asoses

    assert "time" in my_asoses["ORD"].coords
    assert "time" in my_asoses["MDW"].coords


def test_asos_temperature_conversion():
    time_window = [
        datetime(2020, 2, 4, 2, 0),
        datetime(2020, 2, 4, 4, 0),
    ]

    ds = act.discovery.get_asos_data(
        time_window,
        station="ORD",
    )["ORD"]

    expected = (ds["tmpf"] - 32.0) * (5.0 / 9.0)

    np.testing.assert_allclose(
        ds["temp"].values,
        expected.values,
        equal_nan=True,
    )


def test_asos_dewpoint_conversion():
    time_window = [
        datetime(2020, 2, 4, 2, 0),
        datetime(2020, 2, 4, 4, 0),
    ]

    ds = act.discovery.get_asos_data(
        time_window,
        station="ORD",
    )["ORD"]

    expected = (ds["dwpf"] - 32.0) * (5.0 / 9.0)

    np.testing.assert_allclose(
        ds["dwpc"].values,
        expected.values,
        equal_nan=True,
    )


def test_asos_wind_speed_conversion():
    time_window = [
        datetime(2020, 2, 4, 2, 0),
        datetime(2020, 2, 4, 4, 0),
    ]

    ds = act.discovery.get_asos_data(
        time_window,
        station="ORD",
    )["ORD"]

    expected = ds["sknt"] * 0.514444

    np.testing.assert_allclose(
        ds["spdms"].values,
        expected.values,
        equal_nan=True,
    )
