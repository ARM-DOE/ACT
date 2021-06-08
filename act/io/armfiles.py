"""
This module contains I/O operations for loading files that were created for the
Atmospheric Radiation Measurement program supported by the Department of Energy
Office of Science.

"""

import glob
import xarray as xr
import numpy as np
import urllib
import json
import copy
import act.utils as utils
import warnings
from pathlib import Path
import re


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
    return_None : bool, optional
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
        ds = xr.open_mfdataset(filenames, **kwargs)

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
            ds = xr.open_mfdataset(filenames, **kwargs)

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
            if (cftime_to_datetime64 and 'time' in ds.dims and
                    type(ds[var_name].values[0]).__module__.startswith('cftime.')):
                # If we just convert time to datetime64 the group, sel, and other Xarray
                # methods will not work correctly because time is not indexed. Need to
                # use the formation of a Dataset to correctly set the time indexing.
                temp_ds = xr.Dataset(
                    {var_name: (ds[var_name].dims,
                                ds[var_name].astype(desired_time_precision),
                                ds[var_name].attrs)})
                ds[var_name] = temp_ds[var_name]
                temp_ds.close()

                # If time_offset is in file try to convert base_time as well
                if var_name == 'time_offset':
                    ds['base_time'].values = \
                        ds['base_time'].values.astype(desired_time_precision)
        except KeyError:
            pass

    # Check if "time" variable is not in the netCDF file. If so try to use
    # base_time and time_offset to make time variable. Basically a fix for incorrectly
    # formatted files. May require using decode_times=False to initially read the data.
    if (cftime_to_datetime64 and 'time' in ds.dims and
            'time' not in ds.coords and 'time_offset' in ds.data_vars):
        try:
            ds = ds.rename({'time_offset': 'time'})
            ds = ds.set_coords('time')
            del ds['time'].attrs['units']
        except (KeyError, ValueError):
            pass

    # If "time" is not a datetime64 use base_time to calcualte corect values to datetime64
    # by adding base_time to time_offset. time_offset was renamed to time above.
    if (cftime_to_datetime64 and 'time' in ds.dims and 'base_time' in ds.data_vars and
            not np.issubdtype(ds['time'].values.dtype, np.datetime64) and
            not type(ds['time'].values[0]).__module__.startswith('cftime.')):
        # Use microsecond precision to create time since epoch. Then convert to datetime64
        if ds['base_time'].values == ds['time_offset'].values[0]:
            time = ds['time_offset'].values
        else:
            time = (ds['base_time'].values +
                    ds['time_offset'].values * 1000000.).astype('datetime64[us]')
        # Need to use a new Dataset creation to correctly index time for use with
        # .group and .resample methods in Xarray Datasets.
        temp_ds = xr.Dataset({'time': (ds['time'].dims, time, ds['time'].attrs)})

        ds['time'] = temp_ds['time']
        temp_ds.close()
        for att_name in ['units', 'ancillary_variables']:
            try:
                del ds['time'].attrs[att_name]
            except KeyError:
                pass

    # Adding support for wildcards
    if isinstance(filenames, str):
        filenames = glob.glob(filenames)

    # Get file dates and times that were read in to the object
    filenames.sort()
    for f in filenames:
        f = Path(f).name
        pts = re.match(r"(^[a-zA-Z0-9]+)\.([0-9a-z]{2})\.([\d]{8})\.([\d]{6})\.([a-z]{2,3}$)", f)
        # If Not ARM format, read in first time for info
        if pts is not None:
            pts = pts.groups()
            file_dates.append(pts[2])
            file_times.append(pts[3])
        else:
            if ds['time'].size > 1:
                dummy = ds['time'].values[0]
            else:
                dummy = ds['time'].values
            file_dates.append(utils.numpy_to_arm_date(dummy))
            file_times.append(utils.numpy_to_arm_date(dummy, returnTime=True))

    # Add attributes
    ds.attrs['_file_dates'] = file_dates
    ds.attrs['_file_times'] = file_times
    is_arm_file_flag = check_arm_standards(ds)

    # Ensure that we have _datastream set whether or no there's
    # a datastream attribute already.
    if is_arm_file_flag == 0:
        ds.attrs['_datastream'] = "act_datastream"
    else:
        ds.attrs['_datastream'] = ds.attrs['datastream']

    ds.attrs['_arm_standards_flag'] = is_arm_file_flag

    return ds


def check_arm_standards(ds):
    """
    Checks to see if an xarray dataset conforms to ARM standards.

    Parameters
    ----------
    ds : xarray dataset
        The dataset to check.

    Returns
    -------
    flag : int
        The flag corresponding to whether or not the file conforms
        to ARM standards. Bit packed, so 0 for no, 1 for yes

    """
    the_flag = (1 << 0)
    if 'datastream' not in ds.attrs.keys():
        the_flag = 0

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
    base_url = 'https://pcm.arm.gov/pcm/api/dods/'

    # Get data from DOD api
    with urllib.request.urlopen(base_url + proc) as url:
        data = json.loads(url.read().decode())

    # Check version numbers and alert if requested version in not available
    keys = list(data['versions'].keys())
    if version not in keys:
        warnings.warn(' '.join(['Version:', version,
                      'not available or not specified. Using Version:', keys[-1]]),
                      UserWarning)
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


@xr.register_dataset_accessor('write')
class WriteDataset(object):
    """
    Class for cleaning up Dataset before writing to file.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def write_netcdf(self, cleanup_global_atts=True, cleanup_qc_atts=True,
                     join_char='__', make_copy=True, cf_compliant=False,
                     delete_global_attrs=['qc_standards_version', 'qc_method', 'qc_comment'],
                     FillValue=-9999, cf_convention='CF-1.8', **kwargs):
        """
        This is a wrapper around Dataset.to_netcdf to clean up the Dataset before
        writing to disk. Some things are added to global attributes during ACT reading
        process, and QC variables attributes are modified during QC cleanup process.
        This will modify before writing to disk to better
        match Climate & Forecast standards.

        Parameters
        ----------
        cleanup_global_atts : boolean
            Option to cleanup global attributes by removing any global attribute
            that starts with an underscore.
        cleanup_qc_atts : boolean
            Option to convert attributes that would be written as string array
            to be a single character string. CF 1.7 does not allow string attribures.
            Will use a single space a delimeter between values and join_char to replace
            white space between words.
        join_char : str
            The character sting to use for replacing white spaces between words when converting
            a list of strings to single character string attributes.
        make_copy : boolean
            Make a copy before modifying Dataset to write. For large Datasets this
            may add processing time and memory. If modifying the Dataset is OK
            try setting to False.
        cf_compliant : boolean
            Option to output file with additional attributes to make file Climate & Forecast
            complient. May require runing .clean.cleanup() method on the object to fix other
            issues first. This does the best it can but it may not be truely complient. You
            should read the CF documents and try to make complient before writing to file.
        delete_global_attrs : list
            Optional global attributes to be deleted. Defaults to some standard
            QC attributes that are not needed. Can add more or set to None to not
            remove the attributes.
        FillValue : int, float
            The value to use as a _FillValue in output file. This is used to fix
            issues with how Xarray handles missing_value upon reading. It's confusing
            so not a perfect fix. Set to None to leave Xarray to do what it wants.
            Set to a value to be the value used as _FillValue in the file and data
            array. This should then remove missing_value attribute from the file as well.
        cf_convention : str
            The Climate and Forecast convention string to add to Conventions attribute.
        **kwargs : keywords
            Keywords to pass through to Dataset.to_netcdf()

        Examples
        --------
        .. code-block:: python

        ds_object.write.write_netcdf(path='output.nc')

        """

        if make_copy:
            write_obj = copy.deepcopy(self._obj)
        else:
            write_obj = self._obj

        encoding = {}
        if cleanup_global_atts:
            for attr in list(write_obj.attrs):
                if attr.startswith('_'):
                    del write_obj.attrs[attr]

        if cleanup_qc_atts:
            check_atts = ['flag_meanings', 'flag_assessments']
            for var_name in list(write_obj.data_vars):
                if 'standard_name' not in write_obj[var_name].attrs.keys():
                    continue
                for attr_name in check_atts:
                    try:
                        if isinstance(write_obj[var_name].attrs[attr_name], (list, tuple)):
                            att_values = write_obj[var_name].attrs[attr_name]
                            for ii, att_value in enumerate(att_values):
                                att_values[ii] = att_value.replace(' ', join_char)

                            write_obj[var_name].attrs[attr_name] = ' '.join(att_values)
                    except KeyError:
                        pass

                # Tell .to_netcdf() to not add a _FillValue attribute for
                # quality control variables.
                if FillValue is not None:
                    encoding[var_name] = {'_FillValue': None}

            # Clean up _FillValue vs missing_value mess by creating an
            # encoding dictionary with each variable's _FillValue set to
            # requested fill value. May need to improve upon this for data type
            # and other issues in the future.
            if FillValue is not None:
                skip_variables = (['base_time', 'time_offset', 'qc_time'] +
                                  list(encoding.keys()))
                for var_name in list(write_obj.data_vars):
                    if var_name not in skip_variables:
                        encoding[var_name] = {'_FillValue': FillValue}

        if delete_global_attrs is not None:
            for attr in delete_global_attrs:
                try:
                    del write_obj.attrs[attr]
                except KeyError:
                    pass

        # If requested update global attributes and variables attributes for required
        # CF attributes.
        if cf_compliant:
            # Get variable names and standard name for each variable
            var_names = list(write_obj.keys())
            standard_names = []
            for var_name in var_names:
                try:
                    standard_names.append(write_obj[var_name].attrs['standard_name'])
                except KeyError:
                    standard_names.append(None)

            # Check if time varible has axis and standard_name attribute
            coord_name = 'time'
            try:
                write_obj[coord_name].attrs['axis']
            except KeyError:
                try:
                    write_obj[coord_name].attrs['axis'] = 'T'
                except KeyError:
                    pass

            try:
                write_obj[coord_name].attrs['standard_name']
            except KeyError:
                try:
                    write_obj[coord_name].attrs['standard_name'] = 'time'
                except KeyError:
                    pass

            # Try to determine type of dataset by coordinate dimention named time
            # and other factors
            try:
                write_obj.attrs['FeatureType']
            except KeyError:
                dim_names = list(write_obj.dims)
                FeatureType = None
                if dim_names == ['time']:
                    FeatureType = "timeSeries"
                elif len(dim_names) == 2 and 'time' in dim_names and 'bound' in dim_names:
                    FeatureType = "timeSeries"
                elif len(dim_names) >= 2 and 'time' in dim_names:
                    for var_name in var_names:
                        dims = list(write_obj[var_name].dims)
                        if len(dims) == 2 and 'time' in dims:
                            prof_dim = list(set(dims) - set(['time']))[0]
                            if write_obj[prof_dim].values.size > 2:
                                FeatureType = "timeSeriesProfile"
                                break

                if FeatureType is not None:
                    write_obj.attrs['FeatureType'] = FeatureType

            # Add axis and positive attributes to variables with standard_name
            # equal to 'altitude'
            alt_variables = [var_names[ii] for ii, sn in enumerate(standard_names) if sn == 'altitude']
            for var_name in alt_variables:
                try:
                    write_obj[var_name].attrs['axis']
                except KeyError:
                    write_obj[var_name].attrs['axis'] = 'Z'

                try:
                    write_obj[var_name].attrs['positive']
                except KeyError:
                    write_obj[var_name].attrs['positive'] = 'up'

            # Check if the Conventions global attribute lists the CF convention
            try:
                Conventions = write_obj.attrs['Conventions']
                Conventions = Conventions.split()
                cf_listed = False
                for ii in Conventions:
                    if ii.startswith('CF-'):
                        cf_listed = True
                        break
                if not cf_listed:
                    Conventions.append(cf_convention)
                write_obj.attrs['Conventions'] = ' '.join(Conventions)

            except KeyError:
                write_obj.attrs['Conventions'] = str(cf_convention)

            # Reorder global attributes to ensure history is last
            try:
                global_attrs = write_obj.attrs
                history = copy.copy(global_attrs['history'])
                del global_attrs['history']
                global_attrs['history'] = history
            except KeyError:
                pass

        write_obj.to_netcdf(encoding=encoding, **kwargs)
