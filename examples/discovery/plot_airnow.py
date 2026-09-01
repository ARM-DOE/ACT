"""
AirNow Data
-----------

This example shows the different ways to pull air quality information
from EPA's AirNow API for an area near the ARM Southern Great Plains
(SGP) atmospheric observatory.

"""

import os
from datetime import datetime

import matplotlib.pyplot as plt

import act

# You need an account and token from https://docs.airnowapi.org/ first.
token = os.getenv('AIRNOW_API')

if token is not None and len(token) > 0:
    # Get current forecast values for reporting areas within 100 miles of
    # the ZIP code. A latitude/longitude location can also be used instead:
    #
    # results = act.discovery.get_airnow_forecast(
    #     token, date, distance=100, latlon=[41.958, -88.12]
    # )
    #
    # The updated AirNow forecast service provides current forecast
    # information, so use today's date.
    date = datetime.now().strftime('%Y-%m-%d')
    results = act.discovery.get_airnow_forecast(token, date, zipcode=74630, distance=100)

    # The results are returned as a simple xarray Dataset converted from
    # the AirNow tabular response. ACT normalizes the forecast AQI field
    # to "AQI".
    print(results)

    # Historical daily observations are now requested by state rather than
    # by ZIP code or latitude/longitude. This returns daily AQI values for
    # reporting areas across the selected state.
    results = act.discovery.get_airnow_obs(token, date='2025-05-01', state='OK')

    # Historical observations include fields such as DailyAQI and
    # DailyAQICategoryName.
    print(results)

    # Current observations can still be requested using either a ZIP code
    # or latitude/longitude. The updated service returns nearby monitoring
    # sites and includes fields such as NowcastAQI and AqiCategoryName.
    results = act.discovery.get_airnow_obs(token, zipcode=74630, distance=100)

    print(results)

    # This call gets station data for a time period within the provided
    # bounding box. The existing /aq/data/ endpoint is not being retired.
    # The returned object has time as a coordinate and can be used with
    # ACT plotting after reducing the site dimension.
    lat_lon = '-98.172,35.879,-96.76,37.069'
    results = act.discovery.get_airnow_bounded_obs(
        token,
        '2022-05-01T00',
        '2022-05-01T12',
        lat_lon,
        'OZONE,PM25',
        data_type='B',
    )

    # Reduce to a 1D time series for this example.
    results = results.squeeze(dim='sites', drop=False)

    print(results)

    # Plot the available PM2.5 concentration and AQI data.
    display = act.plotting.TimeSeriesDisplay(results)
    display.plot('PM2.5', label='PM2.5')
    display.plot('AQI', label='AQI')

    plt.legend()
    plt.show()
