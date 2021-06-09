"""
Script for downloading data from ARM's Live Data Webservice

"""

import argparse
import json
import sys
import os
try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


def download_data(username, token, datastream,
                  startdate, enddate, time=None, output=None):
    """
    This tool will help users utilize the ARM Live Data Webservice to download
    ARM data.

    Parameters
    ----------
    username : str
        The username to use for logging into the ADC archive.
    token : str
        The access token for accessing the ADC archive.
    datastream : str
        The name of the datastream to acquire.
    startdate : str
        The start date of the data to acquire. Format is YYYY-MM-DD.
    enddate : str
        The end date of the data to acquire. Format is YYYY-MM-DD.
    time: str or None
        The specific time. Format is HHMMSS. Set to None to download all files
        in the given date interval.
    output : str
        The output directory for the data. Set to None to make a folder in the
        current working directory with the same name as *datastream* to place
        the files in.

    Returns
    -------
    files : list
        Returns list of files retrieved

    Notes
    -----
    This programmatic interface allows users to query and automate
    machine-to-machine downloads of ARM data. This tool uses a REST URL and
    specific parameters (saveData, query), user ID and access token, a
    datastream name, a start date, and an end date, and data files matching
    the criteria will be returned to the user and downloaded.

    By using this web service, users can setup cron jobs and automatically
    download data from /data/archive into their workspace. This will also
    eliminate the manual step of following a link in an email to download data.
    All other data files, which are not on the spinning
    disk (on HPSS), will have to go through the regular ordering process.
    More information about this REST API and tools can be found on `ARM Live
    <https://adc.arm.gov/armlive/#scripts>`_.

    To login/register for an access token click `here
    <https://adc.arm.gov/armlive/livedata/home>`_.

    Author: Michael Giansiracusa
    Email: giansiracumt@ornl.gov

    Web Tools Contact: Ranjeet Devarakonda zzr@ornl.gov

    Examples
    --------
    This code will download the netCDF files from the sgpmetE13.b1 datastream
    and place them in a directory named sgpmetE13.b1. The data from 14 Jan to
    20 Jan 2017 will be downloaded. Replace *userName* and *XXXXXXXXXXXXXXXX*
    with your username and token for ARM Data Discovery. See the Notes for
    information on how to obtain a username and token.

    .. code-block:: python

        act.discovery.download_data('userName','XXXXXXXXXXXXXXXX', 'sgpmetE13.b1',
                                    '2017-01-14', '2017-01-20')

    """
    # default start and end are empty
    start, end = '', ''
    # start and end strings for query_url are constructed
    # if the arguments were provided
    if startdate:
        start = "&start={}".format(startdate)
    if enddate:
        end = "&end={}".format(enddate)
    # build the url to query the web service using the arguments provided
    query_url = ('https://adc.arm.gov/armlive/livedata/query?' +
                 'user={0}&ds={1}{2}{3}&wt=json').format(
        ':'.join([username, token]), datastream, start, end)

    # get url response, read the body of the message,
    # and decode from bytes type to utf-8 string
    response_body = urlopen(query_url).read().decode("utf-8")
    # if the response is an html doc, then there was an error with the user
    if response_body[1:14] == "!DOCTYPE html":
        raise ConnectionRefusedError("Error with user. Check username or token.")

    # parse into json object
    response_body_json = json.loads(response_body)

    # construct output directory
    if output:
        # output files to directory specified
        output_dir = os.path.join(output)
    else:
        # if no folder given, add datastream folder
        # to current working dir to prevent file mix-up
        output_dir = os.path.join(os.getcwd(), datastream)

    # not testing, response is successful and files were returned
    if response_body_json is None:
        print("ARM Data Live Webservice does not appear to be functioning")
        return []

    num_files = len(response_body_json["files"])
    file_names = []
    if response_body_json["status"] == "success" and num_files > 0:
        for fname in response_body_json['files']:
            if time is not None:
                if time not in fname:
                    continue
            print("[DOWNLOADING] {}".format(fname))
            # construct link to web service saveData function
            save_data_url = ("https://adc.arm.gov/armlive/livedata/" +
                             "saveData?user={0}&file={1}").format(
                ':'.join([username, token]), fname)
            output_file = os.path.join(output_dir, fname)
            # make directory if it doesn't exist
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            # create file and write bytes to file
            with open(output_file, 'wb') as open_bytes_file:
                open_bytes_file.write(urlopen(save_data_url).read())
            file_names.append(output_file)
    else:
        print("No files returned or url status error.\n"
              "Check datastream name, start, and end date.")

    return file_names
