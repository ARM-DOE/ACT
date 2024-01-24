"""
Function for downloading data from NSF NEON program
using their API.

NEON sites can be found through the NEON website
https://www.neonscience.org/field-sites/explore-field-sites

"""

import requests
import os
import shutil
import pandas as pd


def get_neon_site_products(site_code, print_to_screen=False):
    """
    Returns a list of data products available for a NEON site
    NEON sites can be found through the NEON website
    https://www.neonscience.org/field-sites/explore-field-sites

    Parameters
    ----------
    site : str
        NEON site identifier. Required variable
    print_to_screen : boolean
        If set to True will print to screen

    Returns
    -------
    products : list
        Returns 2D list of data product code and title

    """

    # Every request begins with the server's URL
    server = 'http://data.neonscience.org/api/v0/'

    # Make request, using the sites/ endpoint
    site_request = requests.get(server + 'sites/' + site_code)

    # Convert to Python JSON object
    site_json = site_request.json()

    products = {}
    # View product code and name for every available data product
    for product in site_json['data']['dataProducts']:
        if print_to_screen:
            print(product['dataProductCode'], product['dataProductTitle'])
        products[product['dataProductCode']] = product['dataProductTitle']

    return products


def get_neon_product_avail(site_code, product_code, print_to_screen=False):
    """
    Returns a list of data products available for a NEON site
    NEON sites can be found through the NEON website
    https://www.neonscience.org/field-sites/explore-field-sites

    Parameters
    ----------
    site : str
        NEON site identifier. Required variable
    product_code : str
        NEON product code. Required variable
    print_to_screen : boolean
        If set to True will print to screen

    Returns
    -------
    dates : list
        Returns list of available months of data

    """

    # Every request begins with the server's URL
    server = 'http://data.neonscience.org/api/v0/'

    # Make request, using the sites/ endpoint
    site_request = requests.get(server + 'sites/' + site_code)

    # Convert to Python JSON object
    site_json = site_request.json()

    # View product code and name for every available data product
    for product in site_json['data']['dataProducts']:
        if product['dataProductCode'] != product_code:
            continue
        if print_to_screen:
            print(product['availableMonths'])
        dates = product['availableMonths']

    return dates


def download_neon_data(site_code, product_code, start_date, end_date=None, output_dir=None):
    """
    Returns a list of data products available for a NEON site.  Please be sure to view the
    readme files that are downloaded as well as there may be a number of different products.

    If you want more information on the NEON file formats, please see:
    https://www.neonscience.org/data-samples/data-management/data-formats-conventions

    NEON sites can be found through the NEON website
    https://www.neonscience.org/field-sites/explore-field-sites

    Please be sure to acknowledge and cite the NEON program and data products appropriately:
    https://www.neonscience.org/data-samples/data-policies-citation

    Parameters
    ----------
    site : str
        NEON site identifier. Required variable
    product_code : str
        NEON product code. Required variable
    start_date : str
        Start date of the range to download in YYYY-MM format
    end_date : str
        End date of the range to download in YYYY-MM format.
        If None, will just download data for start_date
    output_dir : str
        Local directory to store the data.  If None, will default to
        [current working directory]/[site_code]_[product_code]

    Returns
    -------
    files : list
        Returns a list of files that were downloaded

    """

    # Every request begins with the server's URL
    server = 'http://data.neonscience.org/api/v0/'

    # Get dates to pass in
    if end_date is not None:
        date_range = pd.date_range(start_date, end_date, freq='MS').strftime('%Y-%m').tolist()
    else:
        date_range = [start_date]

    # For each month, download data for specified site/product
    files = []
    for date in date_range:
        # Make Request
        data_request = requests.get(server + 'data/' + product_code + '/' + site_code + '/' + date)
        data_json = data_request.json()

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), site_code + '_' + product_code)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file in data_json['data']['files']:
            print('[DOWNLOADING] ', file['name'])
            output_filename = os.path.join(output_dir, file['name'])
            with requests.get(file['url'], stream=True) as r:
                with open(output_filename, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                    files.append(output_filename)

    return files
