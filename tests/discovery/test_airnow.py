import os

import numpy as np

import act


def test_get_airnow():
    token = os.getenv('AIRNOW_API')
    if token is not None:
        if len(token) == 0:
            return
        results = act.discovery.get_airnow_forecast(token, '2022-05-01', zipcode=60108, distance=50)
        assert results['CategoryName'].values[0] == 'Good'
        assert results['AQI'].values[2] == -1
        assert results['ReportingArea'].values[3] == 'Aurora and Elgin'

        results = act.discovery.get_airnow_forecast(
            token, '2022-05-01', distance=50, latlon=[41.958, -88.12]
        )
        assert results['CategoryName'].values[3] == 'Good'
        assert results['AQI'].values[2] == -1
        assert results['ReportingArea'][3] == 'Aurora and Elgin'

        results = act.discovery.get_airnow_obs(token, date='2022-05-01', zipcode=60108, distance=50)
        assert results['AQI'].values[0] == 26
        assert results['ParameterName'].values[1] == 'PM2.5'
        assert results['CategoryName'].values[0] == 'Good'

        results = act.discovery.get_airnow_obs(token, zipcode=60108, distance=50)
        assert results['ReportingArea'].values[0] == 'Aurora and Elgin'
        results = act.discovery.get_airnow_obs(token, latlon=[41.958, -88.12], distance=50)
        assert results['StateCode'].values[0] == 'IL'

        with np.testing.assert_raises(NameError):
            results = act.discovery.get_airnow_obs(token)
        with np.testing.assert_raises(NameError):
            results = act.discovery.get_airnow_forecast(token, '2022-05-01')

        results = act.discovery.get_airnow_obs(
            token, date='2022-05-01', distance=50, latlon=[41.958, -88.12]
        )
        assert results['AQI'].values[0] == 26
        assert results['ParameterName'].values[1] == 'PM2.5'
        assert results['CategoryName'].values[0] == 'Good'

        lat_lon = '-88.245401,41.871346,-87.685099,42.234359'
        results = act.discovery.get_airnow_bounded_obs(
            token, '2022-05-01T00', '2022-05-01T12', lat_lon, 'OZONE,PM25', data_type='B'
        )
        assert results['PM2.5'].values[-1, 0] == 1.8
        assert results['OZONE'].values[0, 0] == 37.0
        assert len(results['time'].values) == 13
