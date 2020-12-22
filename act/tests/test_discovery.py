import act
import requests
import numpy as np
import os
import glob
from datetime import datetime
from act.discovery import get_asos
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def test_cropType():
    year = 2018
    lat = 37.1509
    lon = -98.362
    try:
        crop = act.discovery.get_CropScape.croptype(lat, lon, year)
    except Exception:
        return

    if crop is not None:
        assert crop == 'Grass/Pasture'


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


def test_get_armfile():
    if not os.path.isdir((os.getcwd() + '/data/')):
        os.makedirs((os.getcwd() + '/data/'))

    uname = os.getenv('ARM_USERNAME')
    token = os.getenv('ARM_PASSWORD')

    if uname is not None:
        datastream = 'sgpmetE13.b1'
        startdate = '2020-01-01'
        enddate = startdate
        outdir = os.getcwd() + '/data/'

        act.discovery.get_armfiles.download_data(uname, token, datastream,
                                                 startdate, enddate,
                                                 output=outdir)
        files = glob.glob(outdir + datastream + '*20200101*cdf')
        assert files is not None
        assert 'sgpmetE13' in files[0]

        if files is not None:
            os.remove(files[0])
