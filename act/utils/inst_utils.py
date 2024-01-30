"""
Functions containing utilities for instruments.

"""


def decode_present_weather(ds, variable=None, decoded_name=None):
    """
    This function is to decode codes reported from automatic weather stations such as the PWD22.
    This is based on WMO Table 4680 as well as a supplement table for WMO table 4677.

    Parameters
    ----------
    ds : xarray.Dataset
        ACT or Xarray dataset from which to convert codes
    variable : string
        Variable to decode
    decoded_name : string
        New variable name to store updated labels

    Returns
    -------
    ds : xarray.Dataset
        Returns dataset with new decoded data

    References
    ----------
    WMO Manual on Code Volume I.1 A-360.
    https://library.wmo.int/doc_num.php?explnum_id=10235

    """

    # Check to ensure that a variable name is passed
    if variable is None:
        raise ValueError('You must specify a variable')

    if variable not in ds:
        raise ValueError('Variable not in the dataset')

    # Define the weather hash for WMO table 4680.
    weather_4680 = {
        0: 'No significant weather observed',
        1: 'Clouds generally dissolving or becoming less developed during the past hour',
        2: 'State of the sky on the whole unchanged during the past hour',
        3: 'Clouds generally forming or developing during the past hour',
        4: 'Haze or smoke, or dust in suspension in the air, visibility >= 1 km',
        5: 'Haze or smoke, or dust in suspension in the air, visibility < 1 km',
        10: 'Mist',
        11: 'Diamond dust',
        12: 'Distant lightning',
        18: 'Squalls',
        20: 'Fog',
        21: 'Precipitation',
        22: 'Drizzle (not freezing) or snow grains',
        23: 'Rain (not freezing)',
        24: 'Snow',
        25: 'Freezing drizzle or freezing rain',
        26: 'Thunderstorm (with or without precipitation)',
        27: 'Blowing or drifting snow or sand',
        28: 'Blowing or drifting snow or sand, visibility >= 1 km',
        29: 'Blowing or drifting snow or sand, visibility < 1 km',
        30: 'Fog',
        31: 'Fog or ice fog in patches',
        32: 'Fog or ice fog, has become thinner during the past hour',
        33: 'Fog or ice fog, no appreciable change during the past hour',
        34: 'Fog or ice fog, has begun or become thicker during the past hour',
        35: 'Fog, depositing rime',
        40: 'Precipitation',
        41: 'Precipitation, slight or moderate',
        42: 'Precipitation, heavy',
        43: 'Liquid precipitation, slight or moderate',
        44: 'Liquid precipitation, heavy',
        45: 'Solid precipitation, slight or moderate',
        46: 'Solid precipitation, heavy',
        47: 'Freezing precipitation, slight or moderate',
        48: 'Freezing precipitation, heavy',
        50: 'Drizzle',
        51: 'Drizzle, not freezing, slight',
        52: 'Drizzle, not freezing, moderate',
        53: 'Drizzle, not freezing, heavy',
        54: 'Drizzle, freezing, slight',
        55: 'Drizzle, freezing, moderate',
        56: 'Drizzle, freezing, heavy',
        57: 'Drizzle and rain, slight',
        58: 'Drizzle and rain, moderate or heavy',
        60: 'Rain',
        61: 'Rain, not freezing, slight',
        62: 'Rain, not freezing, moderate',
        63: 'Rain, not freezing, heavy',
        64: 'Rain, freezing, slight',
        65: 'Rain, freezing, moderate',
        66: 'Rain, freezing, heavy',
        67: 'Rain (or drizzle) and snow, slight',
        68: 'Rain (or drizzle) and snow, moderate or heavy',
        70: 'Snow',
        71: 'Snow, light',
        72: 'Snow, moderate',
        73: 'Snow, heavy',
        74: 'Ice pellets, slight',
        75: 'Ice pellets, moderate',
        76: 'Ice pellets, heavy',
        77: 'Snow grains',
        78: 'Ice crystals',
        80: 'Shower(s) or Intermittent Precipitation',
        81: 'Rain shower(s) or intermittent rain, slight',
        82: 'Rain shower(s) or intermittent rain, moderate',
        83: 'Rain shower(s) or intermittent rain, heavy',
        84: 'Rain shower(s) or intermittent rain, violent',
        85: 'Snow shower(s) or intermittent snow, slight',
        86: 'Snow shower(s) or intermittent snow, moderate',
        87: 'Snow shower(s) or intermittent snow, heavy',
        89: 'Hail',
        90: 'Thunderstorm',
        91: 'Thunderstorm, slight or moderate, with no precipitation',
        92: 'Thunderstorm, slight or moderate, with rain showers and/or snow showers',
        93: 'Thunderstorm, slight or moderate, with hail',
        94: 'Thunderstorm, heavy, with no precipitation',
        95: 'Thunderstorm, heavy, with rain showers and/or snow showers',
        96: 'Thunderstorm, heavy, with hail',
        99: 'Tornado',
        -9999: 'Missing',
    }

    # Define the weather hash for WMO table 4677.
    weather_4677 = {
        88: 'Shower(s) of snow pellets or small hail, with or without rain or rain and snow mixed, moderate or heavy',
    }

    # Join weather tables
    weather_combined = dict(weather_4680)
    weather_combined.update(weather_4677)

    # Sort keys to be in order
    weather = dict(sorted(weather_combined.items()))

    # If a decoded name is not passed, make one
    if decoded_name is None:
        decoded_name = variable + '_decoded'

    # Get data and fill nans with -9999
    data = ds[variable]
    data = data.fillna(-9999)

    # Get the weather type for each code
    wx_type = [weather[d] for d in data.values]

    # Massage the data array to set back in the dataset
    data.values = wx_type
    if 'long_name' in data.attrs:
        data.attrs['long_name'] = data.attrs['long_name'] + ' Decoded'
    else:
        data.attrs['long_name'] = 'Decoded present weather values'
    if 'valid_min' in data.attrs:
        del data.attrs['valid_min']
    if 'valid_max' in data.attrs:
        del data.attrs['valid_max']

    ds[decoded_name] = data

    return ds
