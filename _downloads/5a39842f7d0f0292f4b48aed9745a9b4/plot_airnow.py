"""
Airnow Data
-----------

This example shows the different ways to pull
air quality information from EPA's AirNow API for
a station near to SGP

"""

import os
import matplotlib.pyplot as plt
import act

# You need an account and token from https://docs.airnowapi.org/ first
token = os.getenv('AIRNOW_API')

if token is not None and len(token) > 0:
    # This first example will get the forcasted values for the date passed
    # at stations within 100 miles of the Zipcode. Can also use latlon instead as
    # results = act.discovery.get_airnow_forecast(token, '2022-05-01', distance=100,
    #                                             latlon=[41.958, -88.12])
    # If the username and token are not set, use the existing sample file
    results = act.discovery.get_airnow_forecast(token, '2022-05-01', zipcode=74630, distance=100)

    # The results show a dataset with air quality information from Oklahoma City
    # The data is not indexed by time and just a rudimentary xarray object from
    # converted from a pandas DataFrame.  Note that the AirNow API labels the data
    # returned as AQI.
    print(results)

    # This call gives the daily average for Ozone, PM2.5 and PM10
    results = act.discovery.get_airnow_obs(token, date='2022-05-01', zipcode=74630, distance=100)
    print(results)

    # This call will get all the station data for a time period within
    # the bounding box provided.  This will return the object with time
    # as a coordinate and can be used with ACT Plotting to plot after
    # squeezing the dimensions.  It can be a 2D time series
    lat_lon = '-98.172,35.879,-96.76,37.069'
    results = act.discovery.get_airnow_bounded_obs(
        token, '2022-05-01T00', '2022-05-01T12', lat_lon, 'OZONE,PM25', data_type='B'
    )
    # Reduce to 1D timeseries
    results = results.squeeze(dim='sites', drop=False)
    print(results)

    # Plot out data but note that Ozone was not return in the results
    display = act.plotting.TimeSeriesDisplay(results)
    display.plot('PM2.5', label='PM2.5')
    display.plot('AQI', label='AQI')
    plt.legend()
    plt.show()
