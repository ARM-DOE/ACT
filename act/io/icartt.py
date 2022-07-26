"""
Modules for Reading/Writing the International Consortium for Atmospheric
Research on Transport and Transformation (ICARTT) file format standards V2.0

References:
    ICARTT V2.0 Standards/Conventions:
    - https://www.earthdata.nasa.gov/s3fs-public/imported/ESDS-RFC-029v2.pdf

"""
import numpy as np
import xarray as xr

import act.utils as utils

import icartt

def hello_world():
    print('hello joe')

def read_icartt(
    filename,
    format=None,
    return_None=False,
    use_cftime=True,
    cftime_to_datetime64=True,
    cleanup_qc=False,
    keep_variables=None,
    **kwargs,
):
    """

    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ICARTT from a single datastream.
    Has some procedures to ensure time is correctly fomatted in returned
    Dataset.

    Parameters
    ----------
    filename : str, pathlib.PosixPath
        Name of file to read.
    format: str
        File Format to Read: FFI 1001 or FFI 2110. Default is 'None'
    return_None : bool, optional
        Catch IOError exception when file not found and return None.
        Default is False.
    ?cleanup_qc : boolean
        Call clean.cleanup() method to convert to standardized ancillary
        quality control variables. This will not allow any keyword options,
        so if non-default behavior is desired will need to call
        clean.cleanup() method on the object after reading the data.
    **kwargs : keywords
        Keywords to pass through to icartt.

    Returns
    -------
    act_obj : Object (or None)
        ACT dataset (or None if no data file(s) found).

    Examples
    --------
    This example will load the example sounding data used for unit testing.

    .. code-block :: python

        import act
        the_ds, the_flag = act.io.armfiles.read_netcdf(
                                act.tests.sample_files.EXAMPLE_SONDE_WILDCARD)
        print(the_ds.attrs._datastream)

    """
    ds = None
    print(filename)
    # Create an exception tuple to use with try statements. Doing it this way
    # so we can add the FileNotFoundError if requested. Can add more error
    # handling in the future.
    except_tuple = (ValueError,)
    if return_None:
        except_tuple = except_tuple + (FileNotFoundError, OSError)

    try:
        # Read data file with ICARTT dataset.
        ict = icartt.Dataset(filename)

    except except_tuple as exception:
            # If requested return None for File not found error
            if type(exception).__name__ == 'FileNotFoundError':
                return None

            # If requested return None for File not found error
            if (type(exception).__name__ == 'OSError'
                and exception.args[0] == 'no files to open'
            ):
                return None

    # Define variables
    var = [x for x in ict.variables]

    # Return Xarray Dataset
    return ict, var
