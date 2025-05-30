"""
Functions specifically for working with QC/DQRs from
the Atmospheric Radiation Measurement Program (ARM).

"""

import datetime as dt
import numpy as np
import requests
import json
import textwrap
from dateutil import parser

from act.config import DEFAULT_DATASTREAM_NAME


def add_dqr_to_qc(
    ds,
    variable=None,
    assessment='incorrect,suspect',
    exclude=None,
    include=None,
    normalize_assessment=True,
    cleanup_qc=True,
    dqr_link=False,
    skip_location_vars=False,
    create_missing_qc_variables=True,
):
    """
    Function to query the ARM DQR web service for reports and
    add as a new quality control test to ancillary quality control
    variable. If no anicllary quality control variable exist a new
    one will be created and lined to the data variable through
    ancillary_variables attribure.

    See online documentation from ARM Data
    Quality Office on the use of the DQR web service.

    https://code.arm.gov/docs/dqrws-examples/wikis/home

    Information about the DQR web-service avaible at
    https://adc.arm.gov/dqrws/

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset
    variable : string, or list of str, or None
        Variables to check DQR web service. If set to None will
        attempt to update all variables.
    assessment : string
        assessment type to get DQRs. Current options include
        'missing', 'suspect', 'incorrect' or any combination separated
        by a comma.
    exclude : list of strings
        DQR IDs to exclude from adding into QC
    include : list of strings
        List of DQR IDs to include in flagging of data. Any other DQR IDs
        will be ignored.
    normalize_assessment : boolean
        The DQR assessment term is different than the embedded QC
        term. Embedded QC uses "Bad" and "Indeterminate" while
        DQRs use "Incorrect" and "Suspect". Setting this will ensure
        the same terms are used for both.
    cleanup_qc : boolean
        Call clean.cleanup() method to convert to standardized ancillary
        quality control variables. Has a little bit of overhead so
        if the Dataset has already been cleaned up, no need to run.
    dqr_link : boolean
        Prints out a link for each DQR to read the full DQR.  Defaults to False
    skip_location_vars : boolean
        Does not apply DQRs to location variables.  This can be useful in the event
        the submitter has erroneously selected all variables.
    create_missing_qc_variables : boolean
        If a quality control variable for the data variable does not exist,
        create the quality control varible and apply DQR.

    Returns
    -------
    ds : xarray.Dataset
        Xarray dataset containing new or updated quality control variables

    Examples
    --------
        .. code-block:: python

            from act.qc.arm import add_dqr_to_qc
            ds = add_dqr_to_qc(ds, variable=['temp_mean', 'atmos_pressure'])


    """

    # DQR Webservice goes off datastreams, pull from the dataset
    if 'datastream' in ds.attrs:
        datastream = ds.attrs['datastream']
    elif '_datastream' in ds.attrs:
        datastream = ds.attrs['_datastream']
    else:
        raise ValueError('Dataset does not have datastream attribute')

    if datastream == DEFAULT_DATASTREAM_NAME:
        raise ValueError(
            "'datastream' name required for DQR service set to default value "
            f"{datastream}. Unable to perform DQR service query."
        )

    # Clean up QC to conform to CF conventions
    if cleanup_qc:
        ds.clean.cleanup()

    # Get time from Dataset
    time = ds['time'].values

    # If the time is not a datetime64 because the read routine was not asked to
    # convert CF variables, convert the time variable for this routine only.
    if not np.issubdtype(time.dtype, np.datetime64):
        units_strings = [
            'seconds since ',
            'minutes since ',
            'hours since ',
            'days since ',
            'milliseconds since ',
            'months since ',
            'years since ',
        ]
        td64_strings = ['s', 'm', 'h', 'D', 'ms', 'M', 'Y']
        units = ds['time'].attrs['units']
        for ii, _ in enumerate(units_strings):
            if units.startswith(units_strings[ii]):
                units = units.replace(units_strings[ii], '')
                td64_string = td64_strings[ii]
                break

        start_time = parser.parse(units)
        start_time = np.datetime64(start_time, td64_string)
        time = start_time + ds['time'].values.astype('timedelta64[s]')

    start_date = time[0].astype('datetime64[s]').astype(dt.datetime).strftime('%Y%m%d')
    end_date = time[-1].astype('datetime64[s]').astype(dt.datetime).strftime('%Y%m%d')

    # Clean up assessment to ensure it is a string with no spaces.
    if isinstance(assessment, (list, tuple)):
        assessment = ','.join(assessment)

    # Not strictly needed but should make things more better.
    assessment = assessment.replace(' ', '')
    assessment = assessment.lower()

    # Create URL
    url = 'https://dqr-web-service.svcs.arm.gov/dqr_full'
    url += f"/{datastream}"
    url += f"/{start_date}/{end_date}"
    url += f"/{assessment}"

    # Call web service
    req = requests.get(url)

    # Check status values and raise error if not successful
    status = req.status_code
    if status == 400:
        raise ValueError('Check parameters')
    if status == 500:
        raise ValueError('DQR Webservice Temporarily Down')

    # Convert from string to dictionary
    docs = json.loads(req.text)

    # If no DQRs found will not have a key with datastream.
    # The status will also be 404.
    try:
        docs = docs[datastream]
    except KeyError:
        return ds

    dqr_results = {}
    for quality_category in docs:
        for dqr_number in docs[quality_category]:
            if exclude is not None and dqr_number in exclude:
                continue

            if include is not None and dqr_number not in include:
                continue

            index = np.array([], dtype=np.int32)
            for time_range in docs[quality_category][dqr_number]['dates']:
                starttime = np.datetime64(time_range['start_date'])
                endtime = np.datetime64(time_range['end_date'])
                ind = np.where((time >= starttime) & (time <= endtime))
                if ind[0].size > 0:
                    index = np.append(index, ind[0])

            if index.size > 0:
                dqr_results[dqr_number] = {
                    'index': index,
                    'test_assessment': quality_category.lower().capitalize(),
                    'test_meaning': f"{dqr_number} : {docs[quality_category][dqr_number]['description']}",
                    'variables': docs[quality_category][dqr_number]['variables'],
                }

            if dqr_link:
                print(
                    f"{dqr_number} - {quality_category.lower().capitalize()}: "
                    f"https://adc.arm.gov/ArchiveServices/DQRService?dqrid={dqr_number}"
                )

    # Check to ensure variable is list
    if variable and not isinstance(variable, (list, tuple)):
        variable = [variable]

    loc_vars = ['lat', 'lon', 'alt', 'latitude', 'longitude', 'altitude']
    for key, value in dqr_results.items():
        for var_name in value['variables']:
            # Do not process on location variables
            if skip_location_vars and var_name in loc_vars:
                continue

            # Do not process time variables
            if var_name in ['time', 'time_offset', 'time_bounds']:
                continue

            # Only process provided variable names
            if variable is not None and var_name not in variable:
                continue

            # Do not process quality control variables as this will create a new
            # quality control variable for the quality control varible.
            try:
                if ds[var_name].attrs['standard_name'] == 'quality_flag':
                    continue
            except KeyError:
                pass

            if (
                create_missing_qc_variables is False
                and ds.qcfilter.check_for_ancillary_qc(var_name, add_if_missing=False) is None
            ):
                continue

            try:
                ds.qcfilter.add_test(
                    var_name,
                    index=np.unique(value['index']),
                    test_meaning=value['test_meaning'],
                    test_assessment=value['test_assessment'],
                )

            except KeyError:  # Variable name not in Dataset
                continue

            except IndexError:
                print(f"Skipping '{var_name}' DQR application because of IndexError")
                continue

            if normalize_assessment:
                ds.clean.normalize_assessment(variables=var_name)

    return ds


def print_dqr(
    datastream,
    start_date,
    end_date,
    variable=None,
    assessment='incorrect,suspect',
    pretty_print=True,
):
    """
    Function to query the ARM DQR web service for reports and
    print the information out in a "pretty" format or in a format
    conducive for pasting into a spreadsheet.

    Information about the DQR web-service avaible at
    https://adc.arm.gov/dqrws/

    Parameters
    ----------
    datastream : string
        Datastream name to query the API for.  Example: sgpmetE13.b1
    start_date : string
        Start date for querying DQR start dates in YYYYMMDD format.  Example: 20240101
    end_date : string
        End date for querying DQR start dates in YYYYMMDD format.  Example: 20240101
    variable : string
        Variable to query DQRs for.  Will only print out DQRs with similar matches.
    assessment : string
        assessment type to get DQRs. Current options include
        'missing', 'suspect', 'incorrect' or any combination separated
        by a comma.
    pretty_print : boolean
        Set to print out in a "pretty" format that is easy to read as default.
        Set to False will print DQR information out in a format that can be pasted
        into a spreadsheet and converted to columns with a comma delimiter

    Returns
    -------
    dqr_results : dict
        Dictionary of DQR information

    """

    # Create URL
    url = 'https://dqr-web-service.svcs.arm.gov/dqr_full'
    url += f"/{datastream}"
    url += f"/{start_date}/{end_date}"
    url += f"/{assessment}"

    # Call web service
    req = requests.get(url)

    # Check status values and raise error if not successful
    status = req.status_code
    if status == 400:
        raise ValueError('Check parameters')
    if status == 404:
        raise ValueError('Check parameters - No DQRs found')
    if status == 500:
        raise ValueError('DQR Webservice Temporarily Down')

    # Convert from string to dictionary
    docs = json.loads(req.text)

    # If no DQRs found will not have a key with datastream.
    # The status will also be 404.
    try:
        docs = docs[datastream]
    except KeyError:
        return

    # Run through results and print out DQR information
    dqr_results = {}
    for quality_category in docs:
        for dqr_number in docs[quality_category]:
            subject = docs[quality_category][dqr_number]['subject']
            description = docs[quality_category][dqr_number]['description']
            suggestions = docs[quality_category][dqr_number]['suggestions']
            variables = docs[quality_category][dqr_number]['variables']
            if variable is not None:
                if variable not in variables:
                    continue
            dates = docs[quality_category][dqr_number]['dates']
            url = 'https://adc.arm.gov/ArchiveServices/DQRService?dqrid=' + dqr_number
            if pretty_print:
                print('\n')
                print(
                    '\t'.join(
                        ['\033[1m' + dqr_number, datastream, quality_category, subject + '\033[0m']
                    )
                )
                print(url)
                print('\n')
                print('\033[1m' + 'Description:' + '\033[0m')
                [print(t) for t in textwrap.wrap(description, width=100)]
                if suggestions is not None:
                    print('\n')
                    print('\033[1m' + 'Suggestions:' + '\033[0m')
                    [print(t) for t in textwrap.wrap(suggestions, width=100)]

                print('\n')
                print('\033[1m' + 'Variables:' + '\033[0m')
                [print('    ', v) for v in variables]
                print('\n')
                print('\033[1m' + 'Dates:' + '\033[0m')
                print('\t', 'Start Date:', '\t\t', 'End Date')
                [print('\t', d['start_date'], '\t', d['end_date']) for d in dates]
                print('-' * 100)
            else:
                if suggestions is None:
                    suggestions = 'None'
                text = ','.join(
                    [dqr_number, datastream, quality_category, subject, description, suggestions]
                )
                text += ',' + ';'.join(variables)
                text += ','
                [print(text + ','.join([d['start_date'], d['end_date']])) for d in dates]

            dqr_results[dqr_number] = {
                'test_assessment': quality_category,
                'subject': subject,
                'description': description,
                'suggestions': suggestions,
                'variables': docs[quality_category][dqr_number]['variables'],
                'dates': dates,
            }

    # Return base information if needed
    return dqr_results
