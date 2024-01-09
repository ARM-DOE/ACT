from datetime import datetime

import numpy as np
import pytest
import pytz

import act


def test_destination_azimuth_distance():
    lat = 37.1509
    lon = -98.362
    lat2, lon2 = act.utils.destination_azimuth_distance(lat, lon, 180.0, 100)

    np.testing.assert_almost_equal(lat2, 37.150, decimal=3)
    np.testing.assert_almost_equal(lon2, -98.361, decimal=3)


def test_add_solar_variable():
    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_NAV)
    new_ds = act.utils.geo_utils.add_solar_variable(ds)

    assert 'sun_variable' in list(new_ds.keys())
    assert new_ds['sun_variable'].values[10] == 1
    assert np.sum(new_ds['sun_variable'].values) >= 598

    new_ds = act.utils.geo_utils.add_solar_variable(ds, dawn_dusk=True)
    assert 'sun_variable' in list(new_ds.keys())
    assert new_ds['sun_variable'].values[10] == 1
    assert np.sum(new_ds['sun_variable'].values) >= 1234

    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET1)
    new_ds = act.utils.geo_utils.add_solar_variable(ds, dawn_dusk=True)
    assert np.sum(new_ds['sun_variable'].values) >= 1046

    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_IRTSST)
    ds = ds.fillna(0)
    new_ds = act.utils.geo_utils.add_solar_variable(ds)
    assert np.sum(new_ds['sun_variable'].values) >= 12

    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_IRTSST)
    ds.drop_vars('lat')
    pytest.raises(ValueError, act.utils.geo_utils.add_solar_variable, ds)

    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_IRTSST)
    ds.drop_vars('lon')
    pytest.raises(ValueError, act.utils.geo_utils.add_solar_variable, ds)
    ds.close()
    new_ds.close()


def test_solar_azimuth_elevation():
    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_NAV)

    elevation, azimuth, distance = act.utils.geo_utils.get_solar_azimuth_elevation(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        time=ds['time'].values,
        library='skyfield',
        temperature_C='standard',
        pressure_mbar='standard',
    )
    assert np.isclose(np.nanmean(elevation), 10.5648, atol=0.001)
    assert np.isclose(np.nanmean(azimuth), 232.0655, atol=0.001)
    assert np.isclose(np.nanmean(distance), 0.985, atol=0.001)


def test_get_sunrise_sunset_noon():
    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_NAV)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        date=ds['time'].values[0],
        library='skyfield',
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        date=ds['time'].values[0],
        library='skyfield',
        timezone=True,
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32, tzinfo=pytz.UTC)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4, tzinfo=pytz.UTC)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10, tzinfo=pytz.UTC)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        date='20180201',
        library='skyfield',
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        date=['20180201'],
        library='skyfield',
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        date=datetime(2018, 2, 1),
        library='skyfield',
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        date=datetime(2018, 2, 1, tzinfo=pytz.UTC),
        library='skyfield',
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=ds['lat'].values[0],
        longitude=ds['lon'].values[0],
        date=[datetime(2018, 2, 1)],
        library='skyfield',
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 1, 31, 22, 36, 32)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 2, 1, 17, 24, 4)
    assert noon[0].replace(microsecond=0) == datetime(2018, 2, 1, 8, 2, 10)

    sunrise, sunset, noon = act.utils.geo_utils.get_sunrise_sunset_noon(
        latitude=85.0, longitude=-140.0, date=[datetime(2018, 6, 1)], library='skyfield'
    )
    assert sunrise[0].replace(microsecond=0) == datetime(2018, 3, 30, 10, 48, 48)
    assert sunset[0].replace(microsecond=0) == datetime(2018, 9, 12, 8, 50, 14)
    assert noon[0].replace(microsecond=0) == datetime(2018, 6, 1, 21, 17, 52)


def test_is_sun_visible():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    result = act.utils.geo_utils.is_sun_visible(
        latitude=ds['lat'].values,
        longitude=ds['lon'].values,
        date_time=ds['time'].values,
    )
    assert len(result) == 48
    assert sum(result) == 20

    result = act.utils.geo_utils.is_sun_visible(
        latitude=ds['lat'].values,
        longitude=ds['lon'].values,
        date_time=ds['time'].values[0],
    )
    assert result == [False]

    result = act.utils.geo_utils.is_sun_visible(
        latitude=ds['lat'].values,
        longitude=ds['lon'].values,
        date_time=[datetime(2019, 11, 25, 13, 30, 00)],
    )
    assert result == [True]

    result = act.utils.geo_utils.is_sun_visible(
        latitude=ds['lat'].values,
        longitude=ds['lon'].values,
        date_time=datetime(2019, 11, 25, 13, 30, 00),
    )
    assert result == [True]
