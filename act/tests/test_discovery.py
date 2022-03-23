import glob
import os
import numpy as np
import requests
from datetime import datetime
from requests.packages.urllib3.exceptions import InsecureRequestWarning

import act
from act.discovery import get_asos

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def test_cropType():
    year = 2018
    lat = 37.15
    lon = -98.362
    # Try for when the cropscape API is not working
    try:
        crop = act.discovery.get_cropscape.croptype(lat, lon, year)
        crop2 = act.discovery.get_cropscape.croptype(lat, lon)
    except Exception:
        return

    if crop is not None:
        assert crop == 'Dbl Crop WinWht/Sorghum'
    if crop2 is not None:
        assert crop2 == 'Sorghum'


def test_get_ord():
    time_window = [datetime(2020, 2, 4, 2, 0), datetime(2020, 2, 12, 10, 0)]
    my_asoses = get_asos(time_window, station='ORD')
    assert 'ORD' in my_asoses.keys()
    assert np.all(
        np.equal(
            my_asoses['ORD']['sknt'].values[:10],
            np.array([13.0, 11.0, 14.0, 14.0, 13.0, 11.0, 14.0, 13.0, 13.0, 13.0]),
        )
    )


def test_get_region():
    my_keys = ['MDW', 'IGQ', 'ORD', '06C', 'PWK', 'LOT', 'GYY']
    time_window = [datetime(2020, 2, 4, 2, 0), datetime(2020, 2, 12, 10, 0)]
    lat_window = (41.8781 - 0.5, 41.8781 + 0.5)
    lon_window = (-87.6298 - 0.5, -87.6298 + 0.5)
    my_asoses = get_asos(time_window, lat_range=lat_window, lon_range=lon_window)
    asos_keys = [x for x in my_asoses.keys()]
    assert asos_keys == my_keys


def test_get_armfile():
    if not os.path.isdir(os.getcwd() + '/data/'):
        os.makedirs(os.getcwd() + '/data/')

    # Place your username and token here
    username = os.getenv('ARM_USERNAME')
    token = os.getenv('ARM_PASSWORD')

    if username is not None and token is not None:
        if len(username) == 0 and len(token) == 0:
            return
        datastream = 'sgpmetE13.b1'
        startdate = '2020-01-01'
        enddate = startdate
        outdir = os.getcwd() + '/data/'

        results = act.discovery.get_armfiles.download_data(
            username, token, datastream, startdate, enddate, output=outdir
        )
        files = glob.glob(outdir + datastream + '*20200101*cdf')
        if len(results) > 0:
            assert files is not None
            assert 'sgpmetE13' in files[0]

        if files is not None:
            if len(files) > 0:
                os.remove(files[0])

        datastream = 'sgpmeetE13.b1'
        act.discovery.get_armfiles.download_data(
            username, token, datastream, startdate, enddate, output=outdir
        )
        files = glob.glob(outdir + datastream + '*20200101*cdf')
        assert len(files) == 0
