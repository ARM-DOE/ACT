"""
Modules for Reading/Writing the International Consortium for Atmospheric
Research on Transport and Transformation (ICARTT) file format standards V2.0

References:
    ICARTT V2.0 Standards/Conventions:
    - https://www.earthdata.nasa.gov/s3fs-public/imported/ESDS-RFC-029v2.pdf

"""
import xarray as xr

try:
    import icartt

    _ICARTT_AVAILABLE = True
    _format = icartt.Formats.FFI1001
except ImportError:
    _ICARTT_AVAILABLE = False
    _format = None


def read_icartt(filename, format=_format, return_None=False, **kwargs):
    """

    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ICARTT from a single datastream. Has some procedures to ensure
    time is correctly fomatted in returned Dataset.

    Parameters
    ----------
    filename : str
        Name of file to read.
    format : str
        ICARTT Format to Read: FFI1001 or FFI2110.
    return_None : bool, optional
        Catch IOError exception when file not found and return None.
        Default is False.
    **kwargs : keywords
        keywords to pass on through to icartt.Dataset.

    Returns
    -------
    ds : xarray.Dataset (or None)
        ACT Xarray dataset (or None if no data file(s) found).

    Examples
    --------
    This example will load the example sounding data used for unit testing.

    .. code-block :: python

        import act
        ds = act.io.icartt.read_icartt(act.tests.sample_files.AAF_SAMPLE_FILE)
        print(ds.attrs['_datastream'])

    """
    if not _ICARTT_AVAILABLE:
        raise ImportError("ICARTT is required to use to read ICARTT files but is not installed")

    ds = None

    # Create an exception tuple to use with try statements. Doing it this way
    # so we can add the FileNotFoundError if requested. Can add more error
    # handling in the future.
    except_tuple = (ValueError,)
    if return_None:
        except_tuple = except_tuple + (FileNotFoundError, OSError)

    try:
        # Read data file with ICARTT dataset.
        ict = icartt.Dataset(filename, format=format, **kwargs)

    except except_tuple as exception:
        # If requested return None for File not found error
        if type(exception).__name__ == 'FileNotFoundError':
            return None

        # If requested return None for File not found error
        if type(exception).__name__ == 'OSError' and exception.args[0] == 'no files to open':
            return None

    # Define the Uncertainty for each variable. Note it may not be calculated.
    # If not calculated, assign 'N/A' to the attribute
    uncertainty = ict.normalComments[6].split(':')[1].split(',')

    # Define the Upper and Lower Limit of Detection Flags
    ulod_flag = ict.normalComments[7].split(':')[1]
    ulod_value = ict.normalComments[8].split(':')[1].split(',')
    llod_flag = ict.normalComments[9].split(':')[1]
    llod_value = ict.normalComments[10].split(':')[1].split(',')

    # Convert ICARTT Object to Xarray Dataset
    ds_container = []
    # Counter for uncertainty/LOD values
    counter = 0

    # Loop over ICART variables, convert to Xarray DataArray, Append.
    for key in ict.variables:
        # Note time is the only independent variable within ICARTT
        # Short name for time must be "Start_UTC" for ICARTT files.
        if key != 'Start_UTC':
            if key == 'qc_flag':
                key2 = 'quality_flag'
            else:
                key2 = key
            da = xr.DataArray(ict.data[key], coords=dict(time=ict.times), name=key2, dims=['time'])
            # Assume if Uncertainity does not match the number of variables,
            # values were not set within the file. Needs to be string!
            if len(uncertainty) != len(ict.variables):
                da.attrs['uncertainty'] = 'N/A'
            else:
                da.attrs['uncertainty'] = uncertainty[counter]

            # Assume if ULOD does not match the number of variables within the
            # the file, ULOD values were not set.
            if len(ulod_value) != len(ict.variables):
                da.attrs['ULOD_Value'] = 'N/A'
            else:
                da.attrs['ULOD_Value'] = ulod_value[counter]

            # Assume if LLOD does not match the number of variables within the
            # the file, LLOD values were not set.
            if len(llod_value) != len(ict.variables):
                da.attrs['LLOD_Value'] = 'N/A'
            else:
                da.attrs['LLOD_Value'] = llod_value[counter]
            # Define the meta data:
            da.attrs['units'] = ict.variables[key].units
            da.attrs['mvc'] = ict.variables[key].miss
            da.attrs['scale_factor'] = ict.variables[key].scale
            da.attrs['ULOD_Flag'] = ulod_flag
            da.attrs['LLOD_Flag'] = llod_flag
            # Append to ds container
            ds_container.append(da.to_dataset(name=key2))
            # up the counter
            counter += 1

    # Concatenate each of the Xarray DataArrays into a single Xarray DataSet
    ds = xr.merge(ds_container)

    # Assign ICARTT Meta data to Xarray DataSet
    ds.attrs['PI'] = ict.PIName
    ds.attrs['PI_Affiliation'] = ict.PIAffiliation
    ds.attrs['Platform'] = ict.dataSourceDescription
    ds.attrs['Mission'] = ict.missionName
    ds.attrs['DateOfCollection'] = ict.dateOfCollection
    ds.attrs['DateOfRevision'] = ict.dateOfRevision
    ds.attrs['Data_Interval'] = ict.dataIntervalCode
    ds.attrs['Independent_Var'] = str(ict.independentVariable)
    ds.attrs['Dependent_Var_Num'] = len(ict.dependentVariables)
    ds.attrs['PI_Contact'] = ict.normalComments[0].split('\n')[0].split(':')[-1]
    ds.attrs['Platform'] = ict.normalComments[1].split(':')[-1]
    ds.attrs['Location'] = ict.normalComments[2].split(':')[-1]
    ds.attrs['Associated_Data'] = ict.normalComments[3].split(':')[-1]
    ds.attrs['Instrument_Info'] = ict.normalComments[4].split(':')[-1]
    ds.attrs['Data_Info'] = ict.normalComments[5][11:]
    ds.attrs['DM_Contact'] = ict.normalComments[11].split(':')[-1]
    ds.attrs['Project_Info'] = ict.normalComments[12].split(':')[-1]
    ds.attrs['Stipulations'] = ict.normalComments[13].split(':')[-1]
    ds.attrs['Comments'] = ict.normalComments[14].split(':')[-1]
    ds.attrs['Revision'] = ict.normalComments[15].split(':')[-1]
    ds.attrs['Revision_Comments'] = ict.normalComments[15 + 1].split(':')[-1]

    # Assign Additional ARM meta data to Xarray DatatSet
    ds.attrs['_datastream'] = filename.split('/')[-1].split('_')[0]

    # Return Xarray Dataset
    return ds
