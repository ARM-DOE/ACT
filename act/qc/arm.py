"""
Functions specifically for working with QC/DQRs from
the Atmospheric Radiation Measurement Program (ARM).

"""

import datetime as dt
import numpy as np
import requests

from act.config import DEFAULT_DATASTREAM_NAME


def add_dqr_to_qc(
    obj,
    variable=None,
    assessment='incorrect,suspect',
    exclude=None,
    include=None,
    normalize_assessment=True,
    cleanup_qc=True,
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
    obj : xarray Dataset
        Data object
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

    Returns
    -------
    obj : xarray Dataset
        Data object

    Examples
    --------
        .. code-block:: python

            from act.qc.arm import add_dqr_to_qc
            obj = add_dqr_to_qc(obj, variable=['temp_mean', 'atmos_pressure'])


    """

    # DQR Webservice goes off datastreams, pull from object
    if 'datastream' in obj.attrs:
        datastream = obj.attrs['datastream']
    elif '_datastream' in obj.attrs:
        datastream = obj.attrs['_datastream']
    else:
        raise ValueError('Object does not have datastream attribute')

    if datastream == DEFAULT_DATASTREAM_NAME:
        raise ValueError("'datastream' name required for DQR service set to default value "
                         f"{datastream}. Unable to perform DQR service query.")

    # Clean up QC to conform to CF conventions
    if cleanup_qc:
        obj.clean.cleanup()

    # In order to properly flag data, get all variables if None. Exclude QC variables.
    if variable is None:
        variable = list(set(obj.data_vars) - set(obj.clean.matched_qc_variables))

    # Check to ensure variable is list
    if not isinstance(variable, (list, tuple)):
        variable = [variable]

    # Loop through each variable and call web service for that variable
    for var_name in variable:
        # Create URL
        url = 'http://www.archive.arm.gov/dqrws/ARMDQR?datastream='
        url += datastream
        url += '&varname=' + var_name
        url += ''.join(
            [
                '&searchmetric=',
                assessment,
                '&dqrfields=dqrid,starttime,endtime,metric,subject',
            ]
        )

        # Call web service
        req = requests.get(url)

        # Check status values and raise error if not successful
        status = req.status_code
        if status == 400:
            raise ValueError('Check parameters')
        if status == 500:
            raise ValueError('DQR Webservice Temporarily Down')

        # Get data and run through each dqr
        dqrs = req.text.splitlines()
        time = obj['time'].values
        dqr_results = {}
        for line in dqrs:
            line = line.split('|')
            dqr_no = line[0]

            # Exclude DQRs if in list
            if exclude is not None and dqr_no in exclude:
                continue

            # Only include if in include list
            if include is not None and dqr_no not in include:
                continue

            starttime = np.datetime64(dt.datetime.utcfromtimestamp(int(line[1])))
            endtime = np.datetime64(dt.datetime.utcfromtimestamp(int(line[2])))
            ind = np.where((time >= starttime) & (time <= endtime))
            if ind[0].size == 0:
                continue

            if dqr_no in dqr_results.keys():
                dqr_results[dqr_no]['index'] = np.append(dqr_results[dqr_no]['index'], ind)
            else:
                dqr_results[dqr_no] = {
                    'index': ind,
                    'test_assessment': line[3],
                    'test_meaning': ': '.join([dqr_no, line[-1]]),
                }

        for key, value in dqr_results.items():
            try:
                obj.qcfilter.add_test(
                    var_name,
                    index=value['index'],
                    test_meaning=value['test_meaning'],
                    test_assessment=value['test_assessment'],
                )
            except IndexError:
                print(f"Skipping '{var_name}' DQR application because of IndexError")

        if normalize_assessment:
            obj.clean.normalize_assessment(variables=var_name)

    return obj
