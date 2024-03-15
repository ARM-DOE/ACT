"""
Function for downloading data from
NOAA Surface Radiation Budget network

"""
from datetime import datetime
import os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


def download_surfrad_data(site=None, startdate=None, enddate=None, output=None):
    """
    Function to download data from the NOAA Surface Radiation Budget network.
    https://gml.noaa.gov/grad/surfrad/

    Parameters
    ----------
    site : str
        3 letter NOAA site identifier. Required variable
        List of sites can be found at https://gml.noaa.gov/grad/surfrad/sitepage.html
    startdate : str
        The start date of the data to acquire. Format is YYYYMMDD. Required variable
    enddate : str
        The end date of the data to acquire. Format is YYYYMMDD
    output : str
        The output directory for the data. Set to None to make a folder in the
        current working directory with the same name as *datastream* to place
        the files in.

    Returns
    -------
    files : list
        Returns list of files retrieved

    """

    if (site is None) or (startdate is None):
        raise ValueError('site and startdate need to be set')

    site = site.lower()
    site_dict = {
        'bnd': 'Bondville_IL',
        'tbl': 'Boulder_CO',
        'dra': 'Desert_Rock_NV',
        'fpk': 'Fort_Peck_MT',
        'gwn': 'Goodwin_Creek_MS',
        'psu': 'Penn_State_PA',
        'sxf': 'Sioux_Falls_SD',
    }
    site_name = site_dict[site]

    # Convert dates to day of year (doy) for NOAA folder structure
    s_doy = datetime.strptime(startdate, '%Y%m%d').timetuple().tm_yday
    year = datetime.strptime(startdate, '%Y%m%d').year
    if enddate is None:
        enddate = startdate
    e_doy = datetime.strptime(enddate, '%Y%m%d').timetuple().tm_yday

    # Set base URL
    url = 'https://gml.noaa.gov/aftp/data/radiation/surfrad/'

    # Construct output directory
    if output:
        # Output files to directory specified
        output_dir = os.path.join(output)
    else:
        # If no folder given, add datastream folder
        # to current working dir to prevent file mix-up
        output_dir = os.path.join(os.getcwd(), site_name + '_surfrad')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set up doy ranges, taking into account changes for a new year
    prev_doy = 0
    if e_doy < s_doy:
        r = list(range(s_doy, 366)) + list(range(1, e_doy + 1))
    else:
        r = list(range(s_doy, e_doy + 1))

    # Set filename variable to return
    filenames = []

    # Loop through each doy in range
    for doy in r:
        # if the previous day is greater than current, assume a new year
        # i.e. 365 -> 001
        if prev_doy > doy:
            year += 1

        # Add filename to url
        file = site + str(year)[2:4] + str(doy) + '.dat'
        new_url = url + site_name + '/' + str(year) + '/' + file

        # Write each file out to a file with same name as online
        output_file = os.path.join(output_dir, file)
        try:
            print('Downloading ' + file)
            with open(output_file, 'wb') as open_bytes_file:
                open_bytes_file.write(urlopen(new_url).read())
            filenames.append(output_file)
        except Exception:
            pass
        prev_doy = doy

    return filenames
