import os
from datetime import datetime

import numpy as np
import pytest

import act


def test_get_airnow():
    token = os.getenv('AIRNOW_API')
    if token is not None:
        if len(token) == 0:
            return

        # The new forecast/current service replaces the separate ZIP code and
        # Lat/Lon forecast endpoints. Use today's date since this service
        # provides current forecast information rather than archived forecasts.
        date = datetime.now().strftime('%Y-%m-%d')

        results = act.discovery.get_airnow_forecast(token, date, zipcode=60108, distance=50)
        assert 'CategoryName' in results
        assert 'AQI' in results
        assert 'ReportingArea' in results

        results = act.discovery.get_airnow_forecast(
            token, date, distance=50, latlon=[41.958, -88.12]
        )
        assert 'CategoryName' in results
        assert 'AQI' in results
        assert 'ReportingArea' in results

        # Current observations now use the combined ZIP/LatLon endpoint.
        results = act.discovery.get_airnow_obs(token, zipcode=60108, distance=50)

        assert 'ReportingAreaName' in results
        assert 'SiteID' in results
        assert 'SiteName' in results
        assert 'NowcastAQI' in results
        assert 'AqiCategoryName' in results

        results = act.discovery.get_airnow_obs(token, latlon=[41.958, -88.12], distance=50)

        assert 'ReportingAreaName' in results
        assert 'SiteID' in results
        assert 'SiteName' in results
        assert 'NowcastAQI' in results
        assert 'AqiCategoryName' in results

        with pytest.raises(NameError):
            results = act.discovery.get_airnow_obs(token)

        with pytest.raises(NameError):
            results = act.discovery.get_airnow_forecast(token, date)

        # Historical observations can no longer be requested by ZIP code or
        # Lat/Lon. AirNow's replacement service is queried by state.
        results = act.discovery.get_airnow_obs(token, date='2025-05-01', state='IL')
        assert 'DailyAQI' in results
        assert 'ParameterName' in results
        assert 'DailyAQICategoryName' in results
        assert 'StateCode' in results
        assert np.all(results['StateCode'].values == 'IL')

        with pytest.raises(NameError):
            results = act.discovery.get_airnow_obs(
                token, date='2025-05-01', latlon=[41.958, -88.12]
            )

        # The hourly monitoring-site bounding-box service is not being retired.
        lat_lon = '-88.245401,41.871346,-87.685099,42.234359'
        results = act.discovery.get_airnow_bounded_obs(
            token, '2022-05-01T00', '2022-05-01T12', lat_lon, 'OZONE,PM25', data_type='B'
        )
        assert results['PM2.5'].values[-1, 0] == 1.8
        assert results['OZONE'].values[0, 0] == 37.0
        assert len(results['time'].values) == 13
