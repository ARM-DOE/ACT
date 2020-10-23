import act
import json
import requests
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
