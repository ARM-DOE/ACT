"""
===============
act.io.armfiles
===============

This module contains I/O operations for loading files that were created for the
Atmospheric Radiation Measurement program supported by the Department of Energy
Office of Science.

"""
# import standard modules
import glob
import xarray as xr
import numpy as np
import urllib
import json
from enum import Flag, auto


class ARMStandardsFlag(Flag):
    """
    This class stores a flag that is returned by
    :ref:act.io.armfiles.check_arm_standards.

    Attributes
    ----------
    OK : flag
        This flag is set if the dataset conforms to ARM standards.
    NO_DATASTREAM : flag
        This flag is set if the dataset does not have a datastream
        field.

    Examples
    --------
    .. code-block:: python

         my_flag = act.io.armfiles.ARMStandardsFlag(
             act.io.armfiles.ARMStandardsFlag.OK)
         assert my_flag.OK

    """
    OK = auto()
    """ The dataset conforms to ARM standards. """
    NO_DATASTREAM = auto()
    """ The dataset does not have a datastream field. """


def read_netcdf(filenames, concat_dim='time', return_None=False,
                combine='by_coords', use_cftime=True, cftime_to_datetime64=True,
                **kwargs):
    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ARM-standard netCDF files from a single datastream. Has some procedures
    to ensure time is correctly fomatted in returned Dataset.

    Parameters
    ----------
    filenames : str or list
        Name of file(s) to read.
    concat_dim : str
        Dimension to concatenate files along. Default value is 'time.'
    return_none : bool, optional
        Catch IOError exception when file not found and return None.
        Default is False.
    combine : str
        String used by xarray.open_mfdataset() to determine how to combine
        data files into one Dataset. See Xarray documentation for options.
        'nested' will remove attributes that differ between files vs.
        'by_coords' which will use the last file's attribute value.
        Default is 'by_coords'.
    use_cftime : boolean
        Option to use cftime library to parse the time units string and correctly
        establish the time values with a units string containing timezone offset.
        This will return the time in cftime format. See cftime_to_datetime64 if
        don't want to convert the times in xarray dataset from cftime to numpy datetime64.
    cftime_to_datetime64 : boolean
        If time is stored as cftime in xarray dataset convert to numpy datetime64. If time
        precision requried is sub millisecond set decode_times=False but leave
        cftime_to_datetime64=True. This will force it to use base_time and time_offset
        to set time.
    **kwargs : keywords
        Keywords to pass through to xarray.open_mfdataset().

    Returns
    -------
    act_obj : Object (or None)
        ACT dataset (or None if no data file(s) found).

    Examples
    --------
    This example will load the example sounding data used for unit testing.

    .. code-block:: python

        import act

        the_ds, the_flag = act.io.armfiles.read_netcdf(
            act.tests.sample_files.EXAMPLE_SONDE_WILDCARD)
        print(the_ds.attrs._datastream)

    """
    file_dates = []
    file_times = []

    # Add funciton keywords to kwargs dictionary for passing into open_mfdataset.
    kwargs['combine'] = combine
    kwargs['concat_dim'] = concat_dim
    kwargs['use_cftime'] = use_cftime

    # Create an exception tuple to use with try statements. Doing it this way
    # so we can add the FileNotFoundError if requested. Can add more error
    # handling in the future.
    except_tuple = (ValueError, )
    if return_None:
        except_tuple = except_tuple + (FileNotFoundError, OSError)

    try:
        # Read data file with Xarray function
        arm_ds = xr.open_mfdataset(filenames, **kwargs)

    except except_tuple as exception:
        # If requested return None for File not found error
        if type(exception).__name__ == 'FileNotFoundError':
            return None

        # If requested return None for File not found error
        if type(exception).__name__ == 'OSError' and exception.args[0] == 'no files to open':
            return None

        # Look at error message and see if could be nested error message. If so
        # update combine keyword and try again. This should allow reading files
        # without a time variable but base_time and time_offset variables.
        if (kwargs['combine'] != 'nested' and type(exception).__name__ == 'ValueError' and
            exception.args[0] == "Could not find any dimension coordinates "
                "to use to order the datasets for concatenation"):
            kwargs['combine'] = 'nested'
            arm_ds = xr.open_mfdataset(filenames, **kwargs)

        else:
            # When all else fails raise the orginal exception
            raise exception

    # Xarray has issues reading a CF formatted time units string if it contains
    # timezone offset without a [+|-] preceeding timezone offset.
    # https://github.com/pydata/xarray/issues/3644
    # To ensure the times are read in correctly need to set use_cftime=True.
    # This will read in time as cftime object. But Xarray uses numpy datetime64
    # natively. This will convert the cftime time values to numpy datetime64. cftime
    # does not preserve the time past ms precision. We will use ms precision for
    # the conversion.
    desired_time_precision = 'datetime64[ms]'
    for var_name in ['time', 'time_offset']:
        try:
            if (cftime_to_datetime64 and 'time' in arm_ds.dims and
                    type(arm_ds[var_name].values[0]).__module__.startswith('cftime.')):
                # If we just convert time to datetime64 the group, sel, and other Xarray
                # methods will not work correctly because time is not indexed. Need to
                # use the formation of a Dataset to correctly set the time indexing.
                temp_ds = xr.Dataset(
                    {var_name: (arm_ds[var_name].dims,
                                arm_ds[var_name].astype(desired_time_precision),
                                arm_ds[var_name].attrs)})
                arm_ds[var_name] = temp_ds[var_name]

                # If time_offset is in file try to convert base_time as well
                if var_name == 'time_offset':
                    arm_ds['base_time'].values = \
                        arm_ds['base_time'].values.astype(desired_time_precision)
        except KeyError:
            pass

    # Check if "time" variable is not in the netCDF file. If so try to use
    # base_time and time_offset to make time variable. Basically a fix for incorrectly
    # formatted files. May require using decode_times=False to initially read the data.
    if (cftime_to_datetime64 and 'time' in arm_ds.dims and
            'time' not in arm_ds.coords and 'time_offset' in arm_ds.data_vars):
        try:
            arm_ds = arm_ds.rename({'time_offset': 'time'})
            arm_ds = arm_ds.set_coords('time')
            del arm_ds['time'].attrs['units']
        except (KeyError, ValueError):
            pass

    # If "time" is not a datetime64 use base_time to calcualte corect values to datetime64
    # by adding base_time to time_offset. time_offset was renamed to time above.
    if (cftime_to_datetime64 and 'time' in arm_ds.dims and 'base_time' in arm_ds.data_vars and
            not np.issubdtype(arm_ds['time'].values.dtype, np.datetime64) and
            not type(arm_ds['time'].values[0]).__module__.startswith('cftime.')):
        # Use microsecond precision to create time since epoch. Then convert to datetime64
        time = (arm_ds['base_time'].values * 1000000 +
                arm_ds['time'].values * 1000000.).astype('datetime64[us]')
        # Need to use a new Dataset creation to correctly index time for use with
        # .group and .resample methods in Xarray Datasets.
        temp_ds = xr.Dataset({'time': (arm_ds['time'].dims, time, arm_ds['time'].attrs)})

        arm_ds['time'] = temp_ds['time']
        for att_name in ['units', 'ancillary_variables']:
            try:
                del arm_ds['time'].attrs[att_name]
            except KeyError:
                pass

    # Adding support for wildcards
    if isinstance(filenames, str):
        filenames = glob.glob(filenames)

    # Get file dates and times that were read in to the object
    filenames.sort()
    for f in filenames:
        file_dates.append(f.split('.')[-3])
        file_times.append(f.split('.')[-2])

    # Add attributes
    arm_ds.attrs['_file_dates'] = file_dates
    arm_ds.attrs['_file_times'] = file_times
    is_arm_file_flag = check_arm_standards(arm_ds)

    # Ensure that we have _datastream set whether or no there's
    # a datastream attribute already.
    if is_arm_file_flag.NO_DATASTREAM is True:
        arm_ds.attrs['_datastream'] = "act_datastream"
    else:
        arm_ds.attrs['_datastream'] = arm_ds.attrs['datastream']

    arm_ds.attrs['_arm_standards_flag'] = is_arm_file_flag

    return arm_ds


def check_arm_standards(ds):
    """
    Checks to see if an xarray dataset conforms to ARM standards.

    Parameters
    ----------
    ds : xarray dataset
        The dataset to check.

    Returns
    -------
    flag : ARMStandardsFlag
        The flag corresponding to whether or not the file conforms
        to ARM standards.

    """
    the_flag = ARMStandardsFlag(ARMStandardsFlag.OK)
    the_flag.NO_DATASTREAM = False
    the_flag.OK = True
    if 'datastream' not in ds.attrs.keys():
        the_flag.OK = False
        the_flag.NO_DATASTREAM = True

    return the_flag


def create_obj_from_arm_dod(proc, set_dims, version='', fill_value=-9999.,
                            scalar_fill_dim=None):
    """
    Queries the ARM DOD api and builds an object based on the ARM DOD and
    the dimension sizes that are passed in.

    Parameters
    ----------
    proc : string
        Process to create the object off of. This is normally in the
        format of inst.level. i.e. vdis.b1 or kazrge.a1
    set_dims : dict
        Dictionary of dims from the DOD and the corresponding sizes.
        Time is required. Code will try and pull from DOD, unless set
        through this variable
        Note: names need to match exactly what is in the dod
        i.e. {'drop_diameter': 50, 'time': 1440}
    version : string
        Version number of the ingest to use. If not set, defaults to
        latest version
    fill_value : float
        Fill value for non-dimension variables. Dimensions cannot have
        duplicate values and are incrementally set (0, 1, 2)
    fill_value : str
        Depending on how the object is set up, sometimes the scalar values
        are dimensioned to the main dimension. i.e. a lat/lon is set to have
        a dimension of time. This is a way to set it up similarly.

    Returns
    -------
    obj : xarray Dataset
        ACT object populated with all variables and attributes.

    Examples
    --------
    .. code-block:: python

        dims = {'time': 1440, 'drop_diameter': 50}
        obj = act.io.armfiles.create_obj_from_arm_dod(
            'vdis.b1', dims, version='1.2', scalar_fill_dim='time')

    """
    # Set base url to get DOD information
    base_url = 'https://pcm.arm.gov/pcmserver/dods/'

    # Get data from DOD api
    with urllib.request.urlopen(base_url + proc) as url:
        data = json.loads(url.read().decode())

    # Check version numbers and alert if requested version in not available
    keys = list(data['versions'].keys())
    if version not in keys:
        print(' '.join(['Version:', version, 'not available or not specified. Using Version:', keys[-1]]))
        version = keys[-1]

    # Create empty xarray dataset
    obj = xr.Dataset()

    # Get the global attributes and add to dataset
    atts = {}
    for a in data['versions'][version]['atts']:
        if a['name'] == 'string':
            continue
        if a['value'] is None:
            a['value'] = ''
        atts[a['name']] = a['value']

    obj.attrs = atts

    # Get variable information and create dataarrays that are
    # then added to the dataset
    # If not passed in through set_dims, will look to the DOD
    # if not set in the DOD, then will raise error
    variables = data['versions'][version]['vars']
    dod_dims = data['versions'][version]['dims']
    for d in dod_dims:
        if d['name'] not in list(set_dims.keys()):
            if d['length'] > 0:
                set_dims[d['name']] = d['length']
            else:
                raise ValueError('Dimension length not set in DOD for ' + d['name'] +
                                 ', nor passed in through set_dim')
    for v in variables:
        dims = v['dims']
        dim_shape = []
        # Using provided dimension data, fill array accordingly for easy overwrite
        if len(dims) == 0:
            if scalar_fill_dim is None:
                data_na = fill_value
            else:
                data_na = np.full(set_dims[scalar_fill_dim], fill_value)
                v['dims'] = scalar_fill_dim
        else:
            for d in dims:
                dim_shape.append(set_dims[d])
            if len(dim_shape) == 1 and v['name'] == dims[0]:
                data_na = np.arange(dim_shape[0])
            else:
                data_na = np.full(dim_shape, fill_value)

        # Get attribute information. Had to do some things to get to print to netcdf
        atts = {}
        str_flag = False
        for a in v['atts']:
            if a['name'] == 'string':
                str_flag = True
                continue
            if a['value'] is None:
                continue
            if str_flag and a['name'] == 'units':
                continue
            atts[a['name']] = a['value']

        da = xr.DataArray(data=data_na, dims=v['dims'], name=v['name'], attrs=atts)
        obj[v['name']] = da

    return obj
