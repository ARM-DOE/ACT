"""
Function for getting CropScape data based on an entered lat/lon.

"""

import datetime
import requests

try:
    from pyproj import Transformer
except ImportError:
    from pyproj.transformer import Transformer


def get_crop_type(lat=None, lon=None, year=None):
    """
    Function for working with the CropScape API to get a crop type based on
    the lat,lon, and year entered. The lat/lon is converted to the projection
    used by CropScape before pased to the API. Note, the requests library
    is indicating a bad handshake with the server so 'verify' is currently
    set to False which is unsecure. Use at your own risk until it can be
    resolved. CropScape - Copyright © Center For Spatial Information Science
    and Systems 2009 - 2018

    Parameters
    ----------
    lat : float
        Latitude of point to retrieve.
    lon : float
        Longitude of point to retrieve.
    year : int
        Year to get croptype for.

    Returns
    -------
    category : string
        String of the crop type at that specific lat/lon for the given year.

    Examples
    --------
    To get the crop type, simply do:

    .. code-block :: python

        type = act.discovery.get_cropscape.croptype(36.8172,-97.1709,'2018')

    """
    # Return if lat/lon are not passed in
    if lat is None or lon is None:
        raise RuntimeError('Lat and Lon need to be provided')

    # Set the CropScape Projection
    outproj = (
        'PROJCS["NAD_1983_Albers",'
        'GEOGCS["NAD83",'
        'DATUM["North_American_Datum_1983",'
        'SPHEROID["GRS 1980",6378137,298.257222101,'
        'AUTHORITY["EPSG","7019"]],'
        'TOWGS84[0,0,0,0,0,0,0],'
        'AUTHORITY["EPSG","6269"]],'
        'PRIMEM["Greenwich",0,'
        'AUTHORITY["EPSG","8901"]],'
        'UNIT["degree",0.0174532925199433,'
        'AUTHORITY["EPSG","9108"]],'
        'AUTHORITY["EPSG","4269"]],'
        'PROJECTION["Albers_Conic_Equal_Area"],'
        'PARAMETER["standard_parallel_1",29.5],'
        'PARAMETER["standard_parallel_2",45.5],'
        'PARAMETER["latitude_of_center",23],'
        'PARAMETER["longitude_of_center",-96],'
        'PARAMETER["false_easting",0],'
        'PARAMETER["false_northing",0],'
        'UNIT["meters",1]]'
    )

    # Set the input projection to be lat/lon
    inproj = 'EPSG:4326'

    # Get the x/y coordinates for CropScape
    transformer = Transformer.from_crs(inproj, outproj)
    x, y = transformer.transform(lat, lon)

    # Build URL
    url = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue?'
    if year is None:
        now = datetime.datetime.now()
        year = now.year - 1

    # Add year, lat, and lon as parameters
    params = {'year': str(year), 'x': str(x), 'y': str(y)}

    # Perform the request.  Note, verify set to False until
    # server SSL errors can be worked out
    try:
        req = requests.get(url, params=params, verify=False, timeout=1)
    except Exception:
        return

    # Return from the webservice is not convertable to json
    # So we need to do some text mining
    text = req.text
    text = text.split(',')
    category = [t for t in text if 'category' in t]
    category = category[0].split(': ')[-1][1:-1]

    return category
