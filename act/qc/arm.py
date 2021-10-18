"""
Functions specifically for working with QC/DQRs from
the Atmospheric Radiation Measurement Program (ARM).

"""

import requests
import datetime as dt
import numpy as np


def add_dqr_to_qc(obj, variable=None, assessment='incorrect,suspect',
                  exclude=None, include=None, normalize_assessment=True,
                  add_qc_variable=None):
    """
    Function to query the ARM DQR web service for reports and
    add as a qc test.  See online documentation from ARM Data
    Quality Office on the use of the DQR web service

    https://code.arm.gov/docs/dqrws-examples/wikis/home

    Parameters
    ----------
    obj : xarray Dataset
        Data object
    variable : string or list
        Variables to check DQR web service for
    assessment : string
        assessment type to get DQRs for
    exclude : list of strings
        DQRs to exclude from adding into QC
    include : list of strings
        List of DQRs to use in flagging of data
    normalize_assessment : boolean
        The DQR assessment term is different than the embedded QC
        term. Embedded QC uses "Bad" and "Indeterminate" while
        DQRs use "Incorrect" and "Suspect". Setting this will ensure
        the same terms are used for both.
    add_qc_variable : string or list
        Variables to add QC information to

    Returns
    -------
    obj : xarray Dataset
        Data object

    """

    # DQR Webservice goes off datastreams, pull from object
    if 'datastream' in obj.attrs:
        datastream = obj.attrs['datastream']
    elif '_datastream' in obj.attrs:
        datastream = obj.attrs['_datastream']
    else:
        raise ValueError('Object does not have datastream attribute')

    # Clean up QC to conform to CF conventions
    obj.clean.cleanup()

    # In order to properly flag data, get all variables if None. Exclude QC variables.
    if variable is None:
        variable = list(set(obj.data_vars) - set(obj.clean.matched_qc_variables))

    # Check to ensure variable is list
    if not isinstance(variable, (list, tuple)):
        variable = [variable]

    # If add_qc_variable is none, set to variables list
    if add_qc_variable is None:
        add_qc_variable = variable

    if not isinstance(add_qc_variable, (list, tuple)):
        add_qc_variable = [add_qc_variable]

    # Loop through each variable and call web service for that variable
    for var_name in variable:
        # Create URL
        url = 'http://www.archive.arm.gov/dqrws/ARMDQR?datastream='
        url += datastream
        url += '&varname=' + var_name
        url += ''.join(['&searchmetric=', assessment,
                        '&dqrfields=dqrid,starttime,endtime,metric,subject'])

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
                dqr_results[dqr_no] = {'index': ind, 'test_assessment': line[3],
                                       'test_meaning': ': '.join([dqr_no, line[-1]])}

        for key, value in dqr_results.items():
            obj.qcfilter.add_test(var_name, index=value['index'],
                                  test_meaning=value['test_meaning'],
                                  test_assessment=value['test_assessment'])

        if normalize_assessment:
            obj.clean.normalize_assessment(variables=var_name)

    return obj
