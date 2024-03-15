"""
Script for downloading ASOS data from the Iowa Mesonet API

"""

import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from io import StringIO

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


def get_asos_data(time_window, lat_range=None, lon_range=None, station=None):
    """
    Returns all of the station observations from the Iowa Mesonet from either
    a given latitude and longitude window or a given station code.

    Parameters
    ----------
    time_window: tuple
        A 2 member list or tuple containing the start and end times. The
        times must be python datetimes.
    lat_range: tuple
        The latitude window to grab all of the ASOS observations from.
    lon_range: tuple
        The longitude window to grab all of the ASOS observations from.
    station: str
        The station ID to grab the ASOS observations from.

    Returns
    -------
    asos_ds: dict of xarray.Datasets
        A dictionary of ACT datasets whose keys are the ASOS station IDs.

    Examples
    --------
    If you want to obtain timeseries of ASOS observations for Chicago O'Hare
    Airport, simply do::

        $ time_window = [datetime(2020, 2, 4, 2, 0), datetime(2020, 2, 10, 10, 0)]
        $ station = "KORD"
        $ my_asoses = act.discovery.get_asos(time_window, station="ORD")
    """
    # First query the database for all of the JSON info for every station
    # Only add stations whose lat/lon are within the Grid's boundaries
    regions = """AF AL_ AI_ AQ_ AG_ AR_ AK AL AM_
        AO_ AS_ AR AW_ AU_ AT_
        AZ_ BA_ BE_ BB_ BG_ BO_ BR_ BF_
        BT_ BS_ BI_ BM_ BB_ BY_ BZ_ BJ_ BW_ AZ CA CA_AB
        CA_BC CD_ CK_ CF_ CG_ CL_ CM_ CO CO_ CN_ CR_ CT
        CU_ CV_ CY_ CZ_ DE DK_ DJ_ DM_ DO_
        DZ EE_ ET_ FK_ FM_ FJ_ FI_ FR_ GF_ PF_
        GA_ GM_ GE_ DE_ GH_ GI_ KY_ GB_ GR_ GL_ GD_
        GU_ GT_ GN_ GW_ GY_ HT_ HN_ HK_ HU_ IS_ IN_
        ID_ IR_ IQ_ IE_ IL_ IT_ CI_ JM_ JP_
        JO_ KZ_ KE_ KI_ KW_ LA_ LV_ LB_ LS_ LR_ LY_
        LT_ LU_ MK_ MG_ MW_ MY_ MV_ ML_ CA_MB
        MH_ MR_ MU_ YT_ MX_ MD_ MC_ MA_ MZ_ MM_ NA_ NP_
        AN_ NL_ CA_NB NC_ CA_NF NF_ NI_
        NE_ NG_ MP_ KP_ CA_NT NO_ CA_NS CA_NU OM_
        CA_ON PK_ PA_ PG_ PY_ PE_ PH_ PN_ PL_
        PT_ CA_PE PR_ QA_ CA_QC RO_ RU_RW_ SH_ KN_
        LC_ VC_ WS_ ST_ CA_SK SA_ SN_ RS_ SC_
        SL_ SG_ SK_ SI_ SB_ SO_ ZA_ KR_ ES_ LK_ SD_ SR_
        SZ_ SE_ CH_ SY_ TW_ TJ_ TZ_ TH_
        TG_ TO_ TT_ TU TN_ TR_ TM_ UG_ UA_ AE_ UN_ UY_
        UZ_ VU_ VE_ VN_ VI_ YE_ CA_YT ZM_ ZW_
        EC_ EG_ FL GA GQ_ HI HR_ IA ID IL IO_ IN KS
        KH_ KY KM_ LA MA MD ME
        MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK
        OR PA RI SC SV_ SD TD_ TN TX UT VA VT VG_
        WA WI WV WY"""

    networks = ['AWOS']
    metadata_list = {}
    if lat_range is not None and lon_range is not None:
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        for region in regions.split():
            networks.append(f'{region}_ASOS')

        site_list = []
        for network in networks:
            # Get metadata
            uri = ('https://mesonet.agron.iastate.edu/' 'geojson/network/%s.geojson') % (network,)
            data = urlopen(uri)
            jdict = json.load(data)
            for site in jdict['features']:
                lat = site['geometry']['coordinates'][1]
                lon = site['geometry']['coordinates'][0]
                if lat >= lat_min and lat <= lat_max:
                    if lon >= lon_min and lon <= lon_max:
                        station_metadata_dict = {}
                        station_metadata_dict['site_latitude'] = lat
                        station_metadata_dict['site_longitude'] = lat
                        for my_keys in site['properties']:
                            station_metadata_dict[my_keys] = site['properties'][my_keys]
                        metadata_list[site['properties']['sid']] = station_metadata_dict
                        site_list.append(site['properties']['sid'])
    elif station is not None:
        site_list = [station]
        for region in regions.split():
            networks.append(f'{region}_ASOS')
        for network in networks:
            # Get metadata
            uri = ('https://mesonet.agron.iastate.edu/' 'geojson/network/%s.geojson') % (network,)
            data = urlopen(uri)
            jdict = json.load(data)
            for site in jdict['features']:
                lat = site['geometry']['coordinates'][1]
                lon = site['geometry']['coordinates'][0]
                if site['properties']['sid'] == station:
                    station_metadata_dict = {}
                    station_metadata_dict['site_latitude'] = lat
                    station_metadata_dict['site_longitude'] = lon
                    for my_keys in site['properties']:
                        if my_keys == 'elevation':
                            station_metadata_dict['elevation'] = (
                                '%f meter' % site['properties'][my_keys]
                            )
                        else:
                            station_metadata_dict[my_keys] = site['properties'][my_keys]
                    metadata_list[station] = station_metadata_dict

        # Get station metadata
    else:
        raise ValueError('Either both lat_range and lon_range or station must ' + 'be specified!')

    # Get the timestamp for each request
    start_time = time_window[0]
    end_time = time_window[1]

    SERVICE = 'http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'
    service = SERVICE + 'data=all&tz=Etc/UTC&format=comma&latlon=yes&'

    service += start_time.strftime('year1=%Y&month1=%m&day1=%d&hour1=%H&minute1=%M&')
    service += end_time.strftime('year2=%Y&month2=%m&day2=%d&hour2=%H&minute2=%M')
    asos_ds = {}
    for stations in site_list:
        uri = f'{service}&station={stations}'
        print(f'Downloading: {stations}')
        data = _download_data(uri)
        buf = StringIO()
        buf.write(data)
        buf.seek(0)

        my_df = pd.read_csv(buf, skiprows=5, na_values='M')

        if len(my_df['lat'].values) == 0:
            warnings.warn(
                'No data available at station %s between time %s and %s'
                % (
                    stations,
                    start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_time.strftime('%Y-%m-%d %H:%M:%S'),
                )
            )
        else:

            def to_datetime(x):
                return datetime.strptime(x, '%Y-%m-%d %H:%M')

            my_df['time'] = my_df['valid'].apply(to_datetime)
            my_df = my_df.set_index('time')
            my_df = my_df.drop('valid', axis=1)
            my_df = my_df.drop('station', axis=1)
            my_df = my_df.to_xarray()

            my_df.attrs = metadata_list[stations]
            my_df['lon'].attrs['units'] = 'degree'
            my_df['lon'].attrs['long_name'] = 'Longitude'
            my_df['lat'].attrs['units'] = 'degree'
            my_df['lat'].attrs['long_name'] = 'Latitude'

            my_df['tmpf'].attrs['units'] = 'degrees Fahrenheit'
            my_df['tmpf'].attrs['long_name'] = 'Temperature in degrees Fahrenheit'

            # Fahrenheit to Celsius
            my_df['temp'] = (5.0 / 9.0 * my_df['tmpf']) - 32.0
            my_df['temp'].attrs['units'] = 'degrees Celsius'
            my_df['temp'].attrs['long_name'] = 'Temperature in degrees Celsius'
            my_df['dwpf'].attrs['units'] = 'degrees Fahrenheit'
            my_df['dwpf'].attrs['long_name'] = 'Dewpoint temperature in degrees Fahrenheit'

            # Fahrenheit to Celsius
            my_df['dwpc'] = (5.0 / 9.0 * my_df['tmpf']) - 32.0
            my_df['dwpc'].attrs['units'] = 'degrees Celsius'
            my_df['dwpc'].attrs['long_name'] = 'Dewpoint temperature in degrees Celsius'
            my_df['relh'].attrs['units'] = 'percent'
            my_df['relh'].attrs['long_name'] = 'Relative humidity'
            my_df['drct'].attrs['units'] = 'degrees'
            my_df['drct'].attrs['long_name'] = 'Wind speed in degrees'
            my_df['sknt'].attrs['units'] = 'knots'
            my_df['sknt'].attrs['long_name'] = 'Wind speed in knots'
            my_df['spdms'] = my_df['sknt'] * 0.514444
            my_df['spdms'].attrs['units'] = 'm s-1'
            my_df['spdms'].attrs['long_name'] = 'Wind speed in meters per second'
            my_df['u'] = -np.sin(np.deg2rad(my_df['drct'])) * my_df['spdms']
            my_df['u'].attrs['units'] = 'm s-1'
            my_df['u'].attrs['long_name'] = 'Zonal component of surface wind'
            my_df['v'] = -np.cos(np.deg2rad(my_df['drct'])) * my_df['spdms']
            my_df['v'].attrs['units'] = 'm s-1'
            my_df['v'].attrs['long_name'] = 'Meridional component of surface wind'
            my_df['mslp'].attrs['units'] = 'mb'
            my_df['mslp'].attrs['long_name'] = 'Mean Sea Level Pressure'
            my_df['alti'].attrs['units'] = 'in Hg'
            my_df['alti'].attrs['long_name'] = 'Atmospheric pressure in inches of Mercury'
            my_df['vsby'].attrs['units'] = 'mi'
            my_df['vsby'].attrs['long_name'] = 'Visibility'
            my_df['vsbykm'] = my_df['vsby'] * 1.60934
            my_df['vsbykm'].attrs['units'] = 'km'
            my_df['vsbykm'].attrs['long_name'] = 'Visibility'
            my_df['gust'] = my_df['gust'] * 0.514444
            my_df['gust'].attrs['units'] = 'm s-1'
            my_df['gust'].attrs['long_name'] = 'Wind gust speed'
            my_df['skyc1'].attrs['long_name'] = 'Sky level 1 coverage'
            my_df['skyc2'].attrs['long_name'] = 'Sky level 2 coverage'
            my_df['skyc3'].attrs['long_name'] = 'Sky level 3 coverage'
            my_df['skyc4'].attrs['long_name'] = 'Sky level 4 coverage'
            my_df['skyl1'] = my_df['skyl1'] * 0.3048
            my_df['skyl2'] = my_df['skyl2'] * 0.3048
            my_df['skyl3'] = my_df['skyl3'] * 0.3048
            my_df['skyl4'] = my_df['skyl4'] * 0.3048
            my_df['skyl1'].attrs['long_name'] = 'Sky level 1 altitude'
            my_df['skyl2'].attrs['long_name'] = 'Sky level 2 altitude'
            my_df['skyl3'].attrs['long_name'] = 'Sky level 3 altitude'
            my_df['skyl4'].attrs['long_name'] = 'Sky level 4 altitude'
            my_df['skyl1'].attrs['long_name'] = 'meter'
            my_df['skyl2'].attrs['long_name'] = 'meter'
            my_df['skyl3'].attrs['long_name'] = 'meter'
            my_df['skyl4'].attrs['long_name'] = 'meter'

            my_df['wxcodes'].attrs['long_name'] = 'Weather code'
            my_df['ice_accretion_1hr'] = my_df['ice_accretion_1hr'] * 2.54
            my_df['ice_accretion_1hr'].attrs['units'] = 'cm'
            my_df['ice_accretion_1hr'].attrs['long_name'] = '1 hour ice accretion'
            my_df['ice_accretion_3hr'] = my_df['ice_accretion_3hr'] * 2.54
            my_df['ice_accretion_3hr'].attrs['units'] = 'cm'
            my_df['ice_accretion_3hr'].attrs['long_name'] = '3 hour ice accretion'
            my_df['ice_accretion_6hr'] = my_df['ice_accretion_3hr'] * 2.54
            my_df['ice_accretion_6hr'].attrs['units'] = 'cm'
            my_df['ice_accretion_6hr'].attrs['long_name'] = '6 hour ice accretion'
            my_df['peak_wind_gust'] = my_df['peak_wind_gust'] * 0.514444
            my_df['peak_wind_gust'].attrs['units'] = 'm s-1'
            my_df['peak_wind_gust'].attrs['long_name'] = 'Peak wind gust speed'
            my_df['peak_wind_drct'].attrs['drct'] = 'degree'
            my_df['peak_wind_drct'].attrs['long_name'] = 'Peak wind gust direction'
            my_df['u_peak'] = -np.sin(np.deg2rad(my_df['peak_wind_drct'])) * my_df['peak_wind_gust']
            my_df['u_peak'].attrs['units'] = 'm s-1'
            my_df['u_peak'].attrs['long_name'] = 'Zonal component of surface wind'
            my_df['v_peak'] = -np.cos(np.deg2rad(my_df['peak_wind_drct'])) * my_df['peak_wind_gust']
            my_df['v_peak'].attrs['units'] = 'm s-1'
            my_df['v_peak'].attrs['long_name'] = 'Meridional component of surface wind'
            my_df['metar'].attrs['long_name'] = 'Raw METAR code'
            my_df.attrs['_datastream'] = stations
            buf.close()

            asos_ds[stations] = my_df
    return asos_ds


def _download_data(uri):
    attempt = 0
    while attempt < 6:
        try:
            data = urlopen(uri, timeout=300).read().decode('utf-8')
            if data is not None and not data.startswith('ERROR'):
                return data
        except Exception as exp:
            print(f'download_data({uri}) failed with {exp}')
            time.sleep(5)
        attempt += 1

    print('Exhausted attempts to download, returning empty data')
    return ''
