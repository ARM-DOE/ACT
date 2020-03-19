"""
act.qc.arm
------------------------------

Functions specifically for working with QC/DQRs from
the Atmospheric Radiation Measurement Program (ARM)

"""

import requests
import datetime as dt
import numpy as np


def add_dqr_to_qc(obj, variable=None, assessment='incorrect,suspect',
                  exclude=None, include=None):
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

    # In order to properly flag data, get all variables if None
    if variable is None:
        variable = [v for v in obj.keys() if 'qc_' not in v]

    # Check to ensure variable is list
    if isinstance(variable, list) is False:
        variable = [variable]

    # Clean up QC to conform to CF conventions
    obj.clean.cleanup()

    # Loop through each variable and call web service for that variable
    for var in variable:
        # Create URL
        url = 'http://www.archive.arm.gov/dqrws/ARMDQR?datastream='
        url += datastream
        url += '&varname=' + var
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

        # Add QC variable
        if 'qc_' + var not in list(obj.keys()):
            obj.qcfilter.create_qc_variable(var)

        # Get data and run through each dqr
        dqrs = req.text.splitlines()
        time = obj['time'].values
        for line in dqrs:
            line = line.split('|')
            # Exclude DQRs if in list
            if exclude is not None:
                if line[0] in exclude:
                    continue

            # Only include if in include list
            if include is not None:
                if line[0] not in include:
                    continue

            line[1] = dt.datetime.fromtimestamp(int(line[1]))
            line[2] = dt.datetime.fromtimestamp(int(line[2]))
            ind = np.where((time >= np.datetime64(line[1])) & (time <= np.datetime64(line[2])))
            if len(ind[0]) == 0:
                continue

            # Add flag to object
            index = sorted(list(ind))
            name = ': '.join([line[0], line[-1]])
            assess = line[3]
            obj.qcfilter.add_test(var, index=index, test_meaning=name,
                                  test_assessment=assess)

    return obj
