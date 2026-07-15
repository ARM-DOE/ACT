"""
Tracking wildfire smoke with AirNow
-----------------------------------

Canadian wildfire smoke routinely degrades air quality across the US
Midwest and Great Lakes. This example uses ACT's AirNow discovery function
``act.discovery.get_airnow_bounded_obs`` as the single data source to build
a picture of one such event, and plots it entirely with ACT's own displays:

* a regional map of the latest hourly surface PM2.5 at every AirNow station
  (``GeographicPlotDisplay``), and
* 24-hour PM2.5 time series for one site each in Duluth MN, Toronto ON, and
  Chicago IL (``TimeSeriesDisplay``).

Because the US EPA AirNow feed also re-serves Canadian provincial monitors
(for example the Ontario Ministry of the Environment network), a single API
path covers stations on both sides of the border.

The date window below is fixed to a July 2026 smoke episode so the example is
reproducible; AirNow serves this range from its rolling archive. Set your free
AirNow token (https://docs.airnowapi.org/) in the ``AIRNOW_API`` environment
variable before running.

Author: Scott Collis
Author: Claude (Anthropic)
"""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import act

# ---------------------------------------------------------------------------
# Fixed date window (UTC). AirNow expects 'YYYY-MM-DDTHH'. Pinning the dates
# (rather than using the current time) keeps the example reproducible.
# ---------------------------------------------------------------------------
START_DATE = '2026-07-14T15'
END_DATE = '2026-07-15T16'

# Free token from https://docs.airnowapi.org/ ; never hard-code it.
token = os.getenv('AIRNOW_API')

# EPA PM2.5 AQI category breakpoints (ug/m^3) for reference on the colorbar.
PM25_MAX = 250.0

if token is not None and len(token) > 0:
    # -----------------------------------------------------------------------
    # 1. Pull every station in the map region through ACT. data_type='B'
    #    returns both AQI and concentrations; mon_type=2 includes permanent
    #    and mobile monitors. The result is a (time, sites) xarray.Dataset.
    # -----------------------------------------------------------------------
    map_bounds = '-104,40,-74,50'
    ds_map = act.discovery.get_airnow_bounded_obs(
        token, START_DATE, END_DATE, map_bounds, 'PM25',
        mon_type=2, data_type='B'
    )

    # Reduce to the latest valid PM2.5 per site. AirNow flags missing as -999.
    pm = ds_map['PM2.5'].values.copy()
    pm[pm < 0] = np.nan
    latest = np.array([
        col[np.where(~np.isnan(col))[0][-1]] if np.any(~np.isnan(col)) else np.nan
        for col in pm.T
    ])

    ds_latest = xr.Dataset(
        {'PM2.5': ('sites', latest)},
        coords={
            'latitude': ('sites', ds_map['latitude'].values),
            'longitude': ('sites', ds_map['longitude'].values),
        },
    )
    ds_latest['PM2.5'].attrs = {'long_name': 'PM2.5', 'units': 'ug/m^3'}

    # -----------------------------------------------------------------------
    # 2. Map the station field with ACT's GeographicPlotDisplay. Passing an
    #    explicit title avoids the display's default time-based title (the
    #    reduced dataset has no time dimension).
    # -----------------------------------------------------------------------
    geo = act.plotting.GeographicPlotDisplay(ds_latest, figsize=(11, 6.5))
    geo.geoplot(
        'PM2.5', lat_field='latitude', lon_field='longitude',
        projection=ccrs.PlateCarree(),
        cartopy_feature=['LAND', 'OCEAN', 'LAKES', 'STATES', 'BORDERS'],
        title='US + Canada surface PM2.5 via AirNow (data pulled with ACT)',
        cmap='turbo', vmin=0, vmax=PM25_MAX,
        marker='o', s=45, edgecolor='white', linewidth=0.3,
    )
    plt.gca().set_extent([-104, -74, 39.5, 50.5], crs=ccrs.PlateCarree())

    # -----------------------------------------------------------------------
    # 3. Pull a tight bounding box around one site in each of three cities and
    #    plot the 24-hour PM2.5 series on a shared y-axis so the magnitudes are
    #    directly comparable.
    # -----------------------------------------------------------------------
    cities = {
        'Duluth, MN': ('-92.3,46.6,-91.9,47.0', 'West Duluth'),
        'Toronto, ON': ('-79.65,43.58,-79.25,43.85', 'Toronto Downtown'),
        'Chicago, IL': ('-87.9,41.7,-87.5,42.05', 'CHI_COM'),
    }

    ds_cities = {}
    for city, (bounds, site) in cities.items():
        ds = act.discovery.get_airnow_bounded_obs(
            token, START_DATE, END_DATE, bounds, 'PM25',
            mon_type=2, data_type='B'
        )
        ds_cities[city] = ds.sel(sites=site)

    tsd = act.plotting.TimeSeriesDisplay(
        ds_cities, subplot_shape=(1, 3), figsize=(14, 3.6)
    )
    for idx, (city, (_, site)) in enumerate(cities.items()):
        tsd.plot(
            'PM2.5', dsname=city, subplot_index=(0, idx),
            y_rng=(0, 600), force_line_plot=True, marker='o',
            set_title=f'{city} \u2014 {site}',
        )
    tsd.fig.tight_layout()

    plt.show()
