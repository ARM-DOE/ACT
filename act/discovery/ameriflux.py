"""
Script for downloading data from Ameriflux's Data Webservice

"""

import os
import requests
import warnings

import pandas as pd

warnings.simplefilter('always')


def download_ameriflux_data(
    user_id,
    user_email,
    site_ids,
    data_product="FLUXNET",
    data_policy=None,
    data_variant="FULLSET",
    agree_policy=True,
    intended_use=None,
    description=None,
    out_dir=None,
    **kwargs,
):
    """
    This tool will allows users to download Ameriflux data. This code is based on the
    original R code found here: https://github.com/chuhousen/amerifluxr

    Parameters
    ----------
    user_id : str
        The user's Ameriflux user ID.
    user_email : str
        The user's email address.
    site_ids : list
        A list of valid site_ids to download data from. List of available sites can be found here:
        https://ameriflux.lbl.gov/sites/site-search/
    data_product : str
        Data product type. Options are BASE-BADM, BIF, or FLUXNET.
        Default is FLUXNET. For more on data products:
        https://ameriflux.lbl.gov/data/flux-data-products/
        https://ameriflux.lbl.gov/data/badm/
    data_policy : str
        Data policy under which data has been licensed by the PI. Options are CCBY4.0 or LEGACY.
    data_variant : str or None
       Variant used for FLUXNET data product. Options are SUBSET, FULLSET, or None.
       Default is FUllSET
    agree_policy : bool
        Acknowledge you read and agree to the AmeriFlux Data use policy. Data policy can be found here:
        https://ameriflux.lbl.gov/data/data-policy/
    intended_use : str
         Planned use for data downloaded. Select best match. Shorter code options are available'
         and when provided will be used to provide full intended use, examples of user codes and what they
         correlate to:
         "synthesis": "Research - Multi-site synthesis",
         "remote_sensing": "Research - Remote sensing",
         "model": "Research - Land model/Earth system model",
         "other_research": "Research - Other",
         "education": "Education (Teacher or Student)",
         "other": "Other"
    description : str
        Brief description of intended use. This will be recorded in the data download log and emailed to site’s PI.
    out_dir : str
        The output directory for the data. Set to None to make a folder in the
        current working directory with the same name as *datastream* to place
        the files in.

    Notes
    -----
    This programmatic interface allows users to query and automate
    machine-to-machine downloads of Ameriflux data. This tool uses a REST URL and
    specific parameters mentioned above and data files matching
    the criteria will be returned to the user and downloaded.

    To login/register for an Ameriflux account:
    https://ameriflux-data.lbl.gov/Pages/RequestAccount.aspx

    Examples
    --------
    This code will download a zip file for BASE-BADM data product at site
    US-A37. See the Notes for information on how to obtain a username and token.

    .. code-block:: python

        act.discovery.download_ameriflux_data(
            user_id, user_email, data_product="BASE-BADM", data_policy="CCBY4.0", data_variant="FULLSET",
            site_ids=["US-A37"], agree_policy=True, intended_use="synthesis",
            description="I intend to use this data for research", out_dir="/home/user/ameriflux_data/",
        )

    Returns
    -------
    files : list
        Returns list of files retrieved

    """
    # Check all inputs are valid
    if not isinstance(user_id, str):
        raise ValueError("user_id should be a string...")

    if not isinstance(user_email, str) or "@" not in user_email:
        raise ValueError("user_email not a valid email...")

    # Check if site_id are valid site IDs
    check_id = _check_site_id(site_ids)
    if isinstance(site_ids, list):
        if any(not valid for valid in check_id):
            warnings.warn(
                f"{', '.join([site_ids[i] for i, valid in enumerate(check_id) if not valid])} not valid AmeriFlux Site ID",
                UserWarning,
            )
            site_ids = [site_ids[i] for i, valid in enumerate(check_id) if valid]
    elif isinstance(site_ids, str):
        if not (check_id or site_ids in ["AA-Flx", "AA-Net"]):
            site_ids = None

    if not site_ids:
        raise ValueError("No valid Site ID in site_ids...")

    # Obtain formal intended use category
    def intended_use_extended(intended_use):
        return {
            "synthesis": "Research - Multi-site synthesis",
            "remote_sensing": "Research - Remote sensing",
            "model": "Research - Land model/Earth system model",
            "other_research": "Research - Other",
            "education": "Education (Teacher or Student)",
            "other": "Other",
        }.get(intended_use)

    if not intended_use_extended(intended_use):
        raise ValueError("Invalid intended_use input...")

    # Check if out_dir is reachable
    if out_dir is None:
        os.makedirs(os.getcwd() + '/data/')
        out_dir = os.getcwd() + '/data/'
    else:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    # Prompt for data policy agreement
    if data_policy == "CCBY4.0":
        warnings.warn(
            "\n"
            "Data use guidelines for AmeriFlux CC-BY-4.0 Data Policy:\n"
            "(1) Data user is free to Share (copy and redistribute the material in any medium or format) and/or Adapt (remix, transform, and build upon the material) for any purpose.\n"
            "(2) Provide a citation to each site data product that includes the data-product DOI and/or recommended publication.\n"
            "(3) Acknowledge funding for supporting AmeriFlux data portal: U.S. Department of Energy Office of Science.\n"
            "\n",
            PolicyWarning,
        )
    elif data_policy == "LEGACY":
        warnings.warn(
            "\n"
            "Data use guidelines for AmeriFlux LEGACY License:\n"
            "(1) When you start in-depth analysis that may result in a publication, contact the data contributors directly, so that they have the opportunity to contribute substantively and become a co-author.\n"
            "(2) Provide a citation to each site data product that includes the data-product DOI.\n"
            "(3) Acknowledge funding for site support if it was provided in the data download information.\n"
            "(4) Acknowledge funding for supporting AmeriFlux data portal: U.S. Department of Energy Office of Science.\n"
            "\n",
            PolicyWarning,
        )
    else:
        raise ValueError("Specify a valid data policy before proceeding...")

    if not agree_policy:
        raise ValueError("Acknowledge data policy before proceeding...")

    if "is_test" in kwargs:
        if kwargs['is_test'] is True:
            test_key = "true"
        else:
            test_key = ""
    else:
        test_key = ""

    # Payload for download web service
    params = {
        "user_id": user_id,
        "user_email": user_email,
        "data_policy": data_policy,
        "data_product": data_product,
        "data_variant": data_variant,
        "site_ids": site_ids,
        "intended_use": intended_use_extended(intended_use),
        "description": f"{description} [Atmospheric data Community Toolkit download]",
        "is_test": test_key,
    }

    result = requests.post(
        _ameriflux_endpoints("data_download"),
        json=params,
        headers={"Content-Type": "application/json"},
    )

    # Check if FTP returns correctly
    if result.status_code == 200:
        link = result.json()
        ftplink = [data_url['url'] for data_url in link.get('data_urls', [])]

        # Check if any site_id has no data
        if not ftplink:
            raise ValueError(f"Cannot find data from {site_ids}")

        # Avoid downloading fluxnet_bif for now
        if (
            isinstance(site_ids, str)
            and site_ids == "AA-Flx"
            and data_policy == "CCBY4.0"
            and len(ftplink) > 1
        ):
            ftplink = [url for url in ftplink if "FLUXNET-BIF" not in url]

        # Get zip file names
        outfname = [os.path.basename(url).split("?")[0] for url in ftplink]

        # Check if any site_ids has no data
        if len(outfname) < len(site_ids):
            miss_site_id = [
                sid for sid in site_ids if sid not in [fname[4:10] for fname in outfname]
            ]
            warnings.warn(f"Cannot find data from {miss_site_id}")

        # Download sequentially
        output_zip_file = [os.path.join(out_dir, fname) for fname in outfname]
        for i, url in enumerate(ftplink):
            response = requests.get(url, stream=True)
            with open(output_zip_file[i], 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Check if downloaded files exist
        miss_download = [i for i, file in enumerate(output_zip_file) if not os.path.exists(file)]
        if miss_download:
            warnings.warn(
                f"Cannot download {[output_zip_file[i] for i in miss_download]} from {[ftplink[i] for i in miss_download]}"
            )

    else:
        raise ValueError("Data download fails, timeout or server error...")

    return output_zip_file


# Return AmeriFlux server endpoints
def _ameriflux_endpoints(endpoint="sitemap"):
    """Retrieves urls for different ameriflux server endpoints. Options include"
    sitemap, site_ccby4, data_year, data_download, and variables"""
    # base urls
    base_url = "https://amfcdn.lbl.gov/"
    api_url = os.path.join(base_url, "api/v1")

    # what to return
    url = {
        "sitemap": os.path.join(api_url, "site_display/AmeriFlux"),
        "site_ccby4": os.path.join(api_url, "site_availability/AmeriFlux/BIF/CCBY4.0"),
        "data_download": os.path.join(api_url, "data_download"),
    }.get(endpoint)
    return url


def _check_site_id(x):
    """Checks if user provided site_ids are valid."""
    response = requests.get(_ameriflux_endpoints("sitemap"))
    df = pd.json_normalize(response.json())
    site_ids = df['SITE_ID'].tolist()
    chk_id = [site_id in site_ids for site_id in x]
    return chk_id


class PolicyWarning(UserWarning):
    pass
