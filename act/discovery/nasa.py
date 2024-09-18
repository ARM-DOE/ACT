"""
Function for downloading data from the NASA Atmospheric Science Data Center
(ASDC), which hosts data including the Atmopsheric Composition
Ground Observation Network.

"""
import os
import requests
import re
import shutil


def download_mplnet_data(
    version=None,
    level=None,
    product=None,
    site=None,
    year=None,
    month=None,
    day=None,
    outdir=None,
):
    """
    Function to download data from the NASA MPL Network Data
    https://mplnet.gsfc.nasa.gov/mplnet_web_services.cgi?download

    Downloaded Products are contained within NETCDF-4, CF compliant files.

    Parameters
    ----------
    version : int
        MPLNet Dataset Version Number (2 or 3).
        All data from 2000 have been processed to Version 3.
        Information on the MPLNet Dataset Version can be found here:
        https://mplnet.gsfc.nasa.gov/versions.htm
    level : int
        MPLNet Product Levels (1, 15, 2).
        MPLNet Levels used to differentiate quality assurance screens.
        Information on the MPLNet Product levels can be found here:
        https://mplnet.gsfc.nasa.gov/product-info/
        Level 1 data should never be used for publication.
    product : str
        MPLNet Product (NRB, CLD, PBL, AER).
            NRB - Lidar signals; volume depolarization ratos, diagnostics
            CLD - Cloud Heights, thin cloud extinction and optical depths, cloud
                    phase
            AER - Aerosol heights; extinction, backscatter, and aerosol
                    depolarization ratio profiles; lidar ratio
            PBL - Surface-Attached Mixed Layer Top and estimated mixed layer AOD
        Information on the MPLNet Products can be found here:
        https://mplnet.gsfc.nasa.gov/product-info/
    year : str
        Four digit Year for desired product download (YYYY).
        Note Level 1 and 1.5 products are available for
        download the day after automated collection.
        Information on the MPLNet naming convention can be found here:
        https://mplnet.gsfc.nasa.gov/product-info/mplnet_file_name.htm
    month : str
        Two digit month for desired product download (MM).
    day : str
        Two digit desired day for product download (DD).
        If day not supplied, will download all data for month supplied
        in a zip file.
    site : str
        MPLNet four letter site identifier.
    outdir : str
        The output directory for the data. Set to None to make a folder in the
        current working directory with the same name as *datastream* to place
        the files in.

    Returns
    -------
    files : list
        Returns list of files retrieved.
    """

    # Generate the data policy agreement information
    print("\nPlease Review the MPLNET Data Policy Prior to Use of MPLNET Data")
    print("The MPLNET Data Policy can be found at:\n\thttps://mplnet.gsfc.nasa.gov/data-policy\n")

    # Generate the data acknowledgement statement, might require site information.
    print(
        "Please Include the Following Acknowledgements in Any Publication \nor"
        + " presentation of MPLNET data, regardless of co-authorship status:"
    )
    print(
        "\n\tThe MPLNET project is funded by the NASA Radiation Sciences Program"
        + " \n\tand Earth Observing System."
    )
    print(
        "\n\tWe thank the MPLNET (PI) for (its/theirs) effort in establishing"
        + " \n\tand maintaining sites.\n"
    )

    # Define the base URL
    base_url = "https://mplnet.gsfc.nasa.gov/download?"

    # Add specific information to the base URL
    if version is None:
        raise ValueError("Please provide a MPLNet Product Version")
    else:
        base_url += "version=V" + str(version)

    if level is None:
        raise ValueError("Please provide a MPLNet Product Level")
    else:
        base_url += "&level=L" + str(level)

    if product is None:
        raise ValueError("Please provide a specific MPLNet Product identifer")
    else:
        base_url += "&product=" + str(product)

    if site is None:
        raise ValueError("Please provide a specific MPLNet site")
    else:
        base_url += "&site=" + str(site)

    if year is None:
        raise ValueError("Year of desired data download is required to download MPLNET data")
    else:
        base_url += "&year=" + str(year)

    if month is None:
        raise ValueError("Month of desired data download is required to download MPLNet data")
    else:
        base_url += "&month=" + str(month)

    if day:
        # Note: Day is not required for the MPLNet download
        base_url += "&day=" + str(day)

    # Construct output directory
    if outdir:
        # Output files to directory specified
        output_dir = os.path.join(outdir)
    else:
        # If no folder given, add MPLNET folder
        # to current working dir to prevent file mix-up
        output_dir = os.path.join(os.getcwd(), "MPLNET")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Make a Request
    files = []
    with requests.get(base_url, stream=True) as r:
        fname = re.findall("filename=(.+)", r.headers['Content-Disposition'])
        # Check for successful file check
        if fname[0][1:-1] == "MPLNET_download_fail.txt":
            raise ValueError(
                "Failed MPLNET Download\n"
                + " File could not be found for the desired input parameters"
                + " for MPLNET Download API"
            )
        else:
            output_filename = os.path.join(output_dir, fname[0][1:-1])
            print("[DOWNLOADING] ", fname[0][1:-1])
            with open(output_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
                files.append(output_filename)

    return files


def get_mplnet_meta(
    sites=None, method=None, year=None, month=None, day=None, print_to_screen=False
):
    """
    Returns a list of meta data from the NASA MPL Network Data
    https://mplnet.gsfc.nasa.gov/mplnet_web_services.cgi?metadata


    Parameters
    ----------
    sites : str
        How to return MPLNET Site Information
            all        - produces output on all sites (active and inactive)
            active     - produces output file containing only active sites
                       (if year, month, or day are not set then uses today's date)
            inactive   - produces output file containing only inactive sites
                       (if year, month, or day are not set then uses today's date)
            planned    - produces output file containing only planned sites
            site_name  - produces output file containing only requested site
            collection - produces output file containing sites in pre-defined
                         collections (e.g. field campaigns or regions)
    year : str
        Four digit Year for desired product download (YYYY).
        Note Level 1 and 1.5 products are available for
        download the day after automated collection.
        Information on the MPLNet naming convention can be found here:
        https://mplnet.gsfc.nasa.gov/product-info/mplnet_file_name.htm
    month : str
        Two digit month for desired product download (MM).
    day : str
        Two digit desired day for product download (DD).
        If day not supplied, will download all data for month supplied
        in a zip file.
    method : str
        Method for returning JSON list of MPLNET GALION format parameters.
            station - returns GALION JSON with only station and PI contact info
            data - return GALION JSON with data elements, station, date and PI
                contact information
    print_to_screen : Boolean
        If true, print MPLNET site identifiers to screen
    """
    # Define the base URL
    base_url = "https://mplnet.gsfc.nasa.gov/operations/sites?api&format=galion"

    if sites is None:
        raise ValueError("Site Parameter is required to download MPLNET Meta Data")
    else:
        base_url += "&sites=" + str(sites)

    if method:
        base_url += "&method=" + str(method)

    if year:
        base_url += "&year=" + str(year)

    if month:
        base_url += "&month=" + str(month)

    if day:
        base_url += "&day=" + str(day)

    with requests.get(base_url, stream=True) as r:
        # Convert to JSON
        site_request = r.json()
        if print_to_screen:
            for i in range(len(site_request)):
                print(site_request[i]['id'])

    return site_request
