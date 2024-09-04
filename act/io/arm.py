"""
This module contains I/O operations for loading files that were created for the
Atmospheric Radiation Measurement program supported by the Department of Energy
Office of Science.
"""

import copy
import datetime as dt
import glob
import json
import re
import tarfile
import tempfile
import urllib
import warnings
from os import PathLike
from pathlib import Path, PosixPath

import numpy as np
import xarray as xr
from cftime import num2date
from netCDF4 import Dataset

import act
import act.utils as utils
from act.config import DEFAULT_DATASTREAM_NAME
from act.utils.io_utils import cleanup_files, is_gunzip_file, unpack_gzip, unpack_tar


def read_arm_netcdf(
    filenames,
    concat_dim=None,
    return_None=False,
    combine='by_coords',
    decode_times=True,
    use_cftime=True,
    use_base_time=False,
    combine_attrs='override',
    cleanup_qc=False,
    keep_variables=None,
    **kwargs,
):
    """

    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ARM-standard netCDF files from a single datastream. Has some procedures
    to ensure time is correctly fomatted in returned Dataset.

    Parameters
    ----------
    filenames : str, pathlib.PosixPath, list of str, list of pathlib.PosixPath
        Name of file(s) to read.
    concat_dim : str
        Dimension to concatenate files along.
    return_None : boolean
        Catch IOError exception when file not found and return None.
        Default is False.
    combine : str
        String used by xarray.open_mfdataset() to determine how to combine
        data files into one Dataset. See Xarray documentation for options.
    decode_times : boolean
        Standard Xarray option to decode time values from int/float to python datetime values.
        Appears the default is to do this anyway but need this option to allow correct usage
        of use_base_time.
    use_cftime : boolean
        Option to use cftime library to parse the time units string and correctly
        establish the time values with a units string containing timezone offset.
        This is used because the Pandas units string parser does not correctly recognize
        time zone offset. Code will automatically detect cftime object and convert to datetime64
        in returned Dataset.
    use_base_time : boolean
        Option to use ARM time variables base_time and time_offset. Useful when the time variable
        is not included (older files) or when the units attribute is incorrectly formatted. Will use
        the values of base_time and time_offset as seconds since epoch and create datetime64 values
        for time coordinate. If set will change decode_times and use_cftime to False.
    combine_attrs : str
        String indicating how to combine attrs of the datasets being merged
    cleanup_qc : boolean
        Call clean.cleanup() method to convert to standardized ancillary quality control
        variables. This will not allow any keyword options, so if non-default behavior is
        desired will need to call clean.cleanup() method on the dataset after reading the data.
    keep_variables : str or list of str
        Variable names to read from data file. Works by creating a list of variable names
        to exclude from reading and passing into open_mfdataset() via drop_variables keyword.
        Still allows use of drop_variables keyword for variables not listed in first file to
        read.
    **kwargs : keywords
        Keywords to pass through to xarray.open_mfdataset().

    Returns
    -------
    ds : xarray.Dataset (or None)
        ACT Xarray dataset (or None if no data file(s) found).

    Examples
    --------
    This example will load the example sounding data used for unit testing.

    .. code-block :: python

        import act
        ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_SONDE_WILDCARD)
        print(ds)

    """

    ds = None
    filenames, cleanup_temp_directory = check_if_tar_gz_file(filenames)

    file_dates = []
    file_times = []

    # If requested to use base_time and time_offset, set keywords to correct attribute values
    # to pass into xarray open_mfdataset(). Need to turn off decode_times and use_cftime
    # or else will try to convert base_time and time_offset. Depending on values of attributes
    # may cause a failure.
    if use_base_time:
        decode_times = False
        use_cftime = False

    # Add funciton keywords to kwargs dictionary for passing into open_mfdataset.
    kwargs['combine'] = combine
    kwargs['concat_dim'] = concat_dim
    kwargs['decode_times'] = decode_times
    kwargs['use_cftime'] = use_cftime
    if len(filenames) > 1 and not isinstance(filenames, str):
        kwargs['combine_attrs'] = combine_attrs

    # Check if keep_variables is set. If so determine correct drop_variables
    if keep_variables is not None:
        drop_variables = None
        if 'drop_variables' in kwargs.keys():
            drop_variables = kwargs['drop_variables']
        kwargs['drop_variables'] = keep_variables_to_drop_variables(
            filenames, keep_variables, drop_variables=drop_variables
        )

    # Create an exception tuple to use with try statements. Doing it this way
    # so we can add the FileNotFoundError if requested. Can add more error
    # handling in the future.
    except_tuple = (ValueError,)
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
        if (
            kwargs['combine'] != 'nested'
            and type(exception).__name__ == 'ValueError'
            and exception.args[0] == 'Could not find any dimension coordinates '
            'to use to order the datasets for concatenation'
        ):
            kwargs['combine'] = 'nested'
            ds = xr.open_mfdataset(filenames, **kwargs)

        else:
            # When all else fails raise the orginal exception
            raise exception

    # If requested use base_time and time_offset to derive time. Assumes that the units
    # of both are in seconds and that the value is number of seconds since epoch.
    if use_base_time:
        time = num2date(
            ds['base_time'].values + ds['time_offset'].values, ds['base_time'].attrs['units']
        )
        time = time.astype('datetime64[ns]')

        # Need to use a new Dataset creation to correctly index time for use with
        # .group and .resample methods in Xarray Datasets.
        temp_ds = xr.Dataset({'time': (ds['time'].dims, time, ds['time'].attrs)})
        ds['time'] = temp_ds['time']
        del temp_ds
        for att_name in ['units', 'ancillary_variables']:
            try:
                del ds['time'].attrs[att_name]
            except KeyError:
                pass

    # Xarray has issues reading a CF formatted time units string if it contains
    # timezone offset without a [+|-] preceeding timezone offset.
    # https://github.com/pydata/xarray/issues/3644
    # To ensure the times are read in correctly need to set use_cftime=True.
    # This will read in time as cftime object. But Xarray uses numpy datetime64
    # natively. This will convert the cftime time values to numpy datetime64.
    desired_time_precision = 'datetime64[ns]'
    for var_name in ['time', 'time_offset']:
        try:
            if 'time' in ds.dims and type(ds[var_name].values[0]).__module__.startswith('cftime.'):
                # If we just convert time to datetime64 the group, sel, and other Xarray
                # methods will not work correctly because time is not indexed. Need to
                # use the formation of a Dataset to correctly set the time indexing.
                temp_ds = xr.Dataset(
                    {
                        var_name: (
                            ds[var_name].dims,
                            ds[var_name].values.astype(desired_time_precision),
                            ds[var_name].attrs,
                        )
                    }
                )
                ds[var_name] = temp_ds[var_name]
                del temp_ds

                # If time_offset is in file try to convert base_time as well
                if var_name == 'time_offset':
                    ds['base_time'].values = ds['base_time'].values.astype(desired_time_precision)
                    ds['base_time'] = ds['base_time'].astype(desired_time_precision)
        except KeyError:
            pass

    # Check if "time" variable is not in the netCDF file. If so try to use
    # base_time and time_offset to make time variable. Basically a fix for incorrectly
    # formatted files. May require using decode_times=False to initially read the data.
    if 'time' in ds.dims and not np.issubdtype(ds['time'].dtype, np.datetime64):
        try:
            ds['time'] = ds['time_offset']
        except (KeyError, ValueError):
            pass

    # Adding support for wildcards
    if isinstance(filenames, str):
        filenames = glob.glob(filenames)
    elif isinstance(filenames, PosixPath):
        filenames = [filenames]

    # Get file dates and times that were read in to the dataset
    filenames.sort()
    for f in filenames:
        f = Path(f).name
        pts = re.match(r'(^[a-zA-Z0-9]+)\.([0-9a-z]{2})\.([\d]{8})\.([\d]{6})\.([a-z]{2,3}$)', f)
        # If Not ARM format, read in first time for info
        if pts is not None:
            pts = pts.groups()
            file_dates.append(pts[2])
            file_times.append(pts[3])
        else:
            if len(ds['time'].shape) > 0:
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
        ds.attrs['_datastream'] = DEFAULT_DATASTREAM_NAME
    else:
        ds.attrs['_datastream'] = ds.attrs['datastream']

    ds.attrs['_arm_standards_flag'] = is_arm_file_flag

    if cleanup_qc:
        ds.clean.cleanup()

    if cleanup_temp_directory:
        cleanup_files(files=filenames)

    return ds


def keep_variables_to_drop_variables(filenames, keep_variables, drop_variables=None):
    """
    Returns a list of variable names to exclude from reading by passing into
    `Xarray.open_dataset` drop_variables keyword. This can greatly help reduce
    loading time and disk space use of the Dataset.

    When passed a netCDF file name, will open the file using the netCDF4 library to get
    list of variable names. There is less overhead reading the varible names using
    netCDF4 library than Xarray. If more than one filename is provided or string is
    used for shell syntax globbing, will use the first file in the list.

    Parameters
    ----------
    filenames : str, pathlib.PosixPath or list of str
        Name of file(s) to read.
    keep_variables : str or list of str
        Variable names desired to keep. Do not need to list associated dimention
        names. These will be automatically kept as well.
    drop_variables : str or list of str
        Variable names to explicitly add to returned list. May be helpful if a variable
        exists in a file that is not in the first file in the list.

    Returns
    -------
    drop_vars : list of str
        Variable names to exclude from returned Dataset by using drop_variables keyword
        when calling Xarray.open_dataset().

    Examples
    --------
    .. code-block :: python

        import act
        filename = '/data/datastream/hou/houkasacrcfrM1.a1/houkasacrcfrM1.a1.20220404.*.nc'
        drop_vars = act.io.arm.keep_variables_to_drop_variables(
            filename, ['lat','lon','alt','crosspolar_differential_phase'],
            drop_variables='variable_name_that_only_exists_in_last_file_of_the_day')

    """
    read_variables = []
    return_variables = []

    if isinstance(keep_variables, str):
        keep_variables = [keep_variables]

    if isinstance(drop_variables, str):
        drop_variables = [drop_variables]

    # If filenames is a list subset to first file name.
    if isinstance(filenames, (list, tuple)):
        filename = filenames[0]
    # If filenames is a string, check if it needs to be expanded in shell
    # first. Then use first returned file name. Else use the string filename.
    elif isinstance(filenames, str):
        filename = glob.glob(filenames)
        if len(filename) == 0:
            return return_variables
        else:
            filename.sort()
            filename = filename[0]

    # Use netCDF4 library to extract the variable and dimension names.
    rootgrp = Dataset(filename, 'r')
    read_variables = list(rootgrp.variables)
    # Loop over the variables to exclude needed coordinate dimention names.
    dims_to_keep = []
    for var_name in keep_variables:
        try:
            dims_to_keep.extend(list(rootgrp[var_name].dimensions))
        except IndexError:
            pass

    rootgrp.close()

    # Remove names not matching keep_varibles excluding the associated coordinate dimentions
    return_variables = set(read_variables) - set(keep_variables) - set(dims_to_keep)

    # Add drop_variables to list
    if drop_variables is not None:
        return_variables = set(return_variables) | set(drop_variables)

    return list(return_variables)


def check_arm_standards(ds):
    """

    Checks to see if an xarray dataset conforms to ARM standards.

    Parameters
    ----------
    ds : Xarray Dataset
        The dataset to check.

    Returns
    -------
    flag : int
        The flag corresponding to whether or not the file conforms
        to ARM standards. Bit packed, so 0 for no, 1 for yes

    """
    the_flag = 1 << 0
    if 'datastream' not in ds.attrs.keys():
        the_flag = 0

    # Check if the historical global attribute name is
    # used instead of updated name of 'datastream'. If so
    # correct the global attributes and flip flag.
    if 'zeb_platform' in ds.attrs.keys():
        ds.attrs['datastream'] = copy.copy(ds.attrs['zeb_platform'])
        del ds.attrs['zeb_platform']
        the_flag = 1 << 0

    return the_flag


def create_ds_from_arm_dod(
    proc, set_dims, version='', fill_value=-9999.0, scalar_fill_dim=None, local_file=False
):
    """

    Queries the ARM DOD api and builds a dataset based on the ARM DOD and
    the dimension sizes that are passed in.

    Parameters
    ----------
    proc : string
        Process to create the dataset off of. This is normally in the
        format of inst.level. i.e. vdis.b1 or kazrge.a1. If local file
        is true, this points to the path of the .dod file.
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
    scalar_fill_dim : str
        Depending on how the dataset is set up, sometimes the scalar values
        are dimensioned to the main dimension. i.e. a lat/lon is set to have
        a dimension of time. This is a way to set it up similarly.
    local_file: bool
        If true, the DOD will be loaded from a file whose name is proc.
        If false, the DOD will be pulled from PCM.
    Returns
    -------
    ds : xarray.Dataset
        ACT Xarray dataset populated with all variables and attributes.

    Examples
    --------
    .. code-block :: python

        dims = {'time': 1440, 'drop_diameter': 50}
        ds = act.io.arm.create_ds_from_arm_dod(
            'vdis.b1', dims, version='1.2', scalar_fill_dim='time')

    """
    # Set base url to get DOD information
    if local_file is False:
        base_url = 'https://pcm.arm.gov/pcm/api/dods/'

        # Get data from DOD api
        with urllib.request.urlopen(base_url + proc) as url:
            data = json.loads(url.read().decode())
    else:
        with open(proc) as file:
            data = json.loads(file.read())

    # Check version numbers and alert if requested version in not available
    keys = list(data['versions'].keys())
    if version not in keys:
        warnings.warn(
            ' '.join(
                ['Version:', version, 'not available or not specified. Using Version:', keys[-1]]
            ),
            UserWarning,
        )
        version = keys[-1]

    # Create empty xarray dataset
    ds = xr.Dataset()

    # Get the global attributes and add to dataset
    atts = {}
    for a in data['versions'][version]['atts']:
        if a['name'] == 'string':
            continue
        if a['value'] is None:
            a['value'] = ''
        atts[a['name']] = a['value']

    ds.attrs = atts

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
                raise ValueError(
                    'Dimension length not set in DOD for '
                    + d['name']
                    + ', nor passed in through set_dim'
                )
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
        ds[v['name']] = da

    return ds


@xr.register_dataset_accessor('write')
class WriteDataset:
    """

    Class for cleaning up Dataset before writing to file.

    """

    def __init__(self, xarray_ds):
        self._ds = xarray_ds

    def write_netcdf(
        self,
        cleanup_global_atts=True,
        cleanup_qc_atts=True,
        join_char='__',
        make_copy=True,
        cf_compliant=False,
        delete_global_attrs=['qc_standards_version', 'qc_method', 'qc_comment'],
        FillValue=True,
        cf_convention='CF-1.8',
        encoding={},
        **kwargs,
    ):
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
            a list of strings to single character string attributes. Main use is with the
            flag_meanings attribute.
        make_copy : boolean
            Make a copy before modifying Dataset to write. For large Datasets this
            may add processing time and memory. If modifying the Dataset is OK
            try setting to False.
        cf_compliant : boolean
            Option to output file with additional attributes to make file Climate & Forecast
            complient. May require runing .clean.cleanup() method on the dataset to fix other
            issues first. This does the best it can but it may not be truely complient. You
            should read the CF documents and try to make complient before writing to file.
        delete_global_attrs : list
            Optional global attributes to be deleted. Defaults to some standard
            QC attributes that are not needed. Can add more or set to None to not
            remove the attributes.
        FillValue : boolean
            Xarray assumes all float type variables had the missing value indicator converted
            to NaN upon reading. to_netcdf() will then write a _FillValue attribute set to NaN.
            Set FillValue to False to supress adding the _FillValue=NaN variable attribute to
            the written file. Set to True to allow to_netcdf() to add the attribute.
            If the Dataset variable already has a _FillValue attribute or a _FillValue key
            is provided in the encoding dictionary those will not be changed and a _FillValue
            will be written to NetCDF file.
        cf_convention : str
            The Climate and Forecast convention string to add to Conventions attribute.
        encoding : dict
            The encoding dictionary used with to_netcdf() method.
        **kwargs : keywords
            Keywords to pass through to Dataset.to_netcdf()

        Examples
        --------
        .. code-block :: python

            ds.write.write_netcdf(path='output.nc')

        """

        if make_copy:
            ds = copy.deepcopy(self._ds)
        else:
            ds = self._ds

        if cleanup_global_atts:
            for attr in list(ds.attrs):
                if attr.startswith('_'):
                    del ds.attrs[attr]

        if cleanup_qc_atts:
            check_atts = ['flag_meanings', 'flag_assessments']
            for var_name in list(ds.data_vars):
                if 'standard_name' not in ds[var_name].attrs.keys():
                    continue

                if ds[var_name].attrs['standard_name'] != "quality_flag":
                    continue

                for attr_name in check_atts:
                    try:
                        att_values = ds[var_name].attrs[attr_name]
                        if isinstance(att_values, (list, tuple)):
                            att_values = [
                                att_value.replace(' ', join_char) for att_value in att_values
                            ]
                            ds[var_name].attrs[attr_name] = ' '.join(att_values)

                    except KeyError:
                        pass

        # Xarray makes an assumption that float type variables were read in and converted
        # missing value indicator to NaN. .to_netcdf() will then automatically assign
        # _FillValue attribute set to NaN when writing. If requested will set _FillValue
        # key in encoding to None which will supress to_netcdf() from adding a _FillValue.
        # If _FillValue attribute or _FillValue key in encoding is already set, will not
        # override and the _FillValue will be written to the file.
        if not FillValue:
            all_var_names = list(ds.coords.keys()) + list(ds.data_vars)
            for var_name in all_var_names:
                if '_FillValue' in ds[var_name].attrs:
                    continue

                if var_name not in encoding.keys():
                    encoding[var_name] = {'_FillValue': None}
                elif '_FillValue' not in encoding[var_name].keys():
                    encoding[var_name]['_FillValue'] = None

        if delete_global_attrs is not None:
            for attr in delete_global_attrs:
                try:
                    del ds.attrs[attr]
                except KeyError:
                    pass

        for var_name in list(ds.keys()):
            if 'string' in list(ds[var_name].attrs.keys()):
                att = ds[var_name].attrs['string']
                ds[var_name].attrs[var_name + '_string'] = att
                del ds[var_name].attrs['string']

        # If requested update global attributes and variables attributes for required
        # CF attributes.
        if cf_compliant:
            # Get variable names and standard name for each variable
            var_names = list(ds.keys())
            standard_names = []
            for var_name in var_names:
                try:
                    standard_names.append(ds[var_name].attrs['standard_name'])
                except KeyError:
                    standard_names.append(None)

            # Check if time varible has axis and standard_name attribute
            coord_name = 'time'
            try:
                ds[coord_name].attrs['axis']
            except KeyError:
                try:
                    ds[coord_name].attrs['axis'] = 'T'
                except KeyError:
                    pass

            try:
                ds[coord_name].attrs['standard_name']
            except KeyError:
                try:
                    ds[coord_name].attrs['standard_name'] = 'time'
                except KeyError:
                    pass

            # Try to determine type of dataset by coordinate dimention named time
            # and other factors
            try:
                ds.attrs['FeatureType']
            except KeyError:
                dim_names = list(ds.dims)
                FeatureType = None
                if dim_names == ['time']:
                    FeatureType = 'timeSeries'
                elif len(dim_names) == 2 and 'time' in dim_names and 'bound' in dim_names:
                    FeatureType = 'timeSeries'
                elif len(dim_names) >= 2 and 'time' in dim_names:
                    for var_name in var_names:
                        dims = list(ds[var_name].dims)
                        if len(dims) == 2 and 'time' in dims:
                            prof_dim = list(set(dims) - {'time'})[0]
                            if ds[prof_dim].values.size > 2:
                                FeatureType = 'timeSeriesProfile'
                                break

                if FeatureType is not None:
                    ds.attrs['FeatureType'] = FeatureType

            # Add axis and positive attributes to variables with standard_name
            # equal to 'altitude'
            alt_variables = [
                var_names[ii] for ii, sn in enumerate(standard_names) if sn == 'altitude'
            ]
            for var_name in alt_variables:
                try:
                    ds[var_name].attrs['axis']
                except KeyError:
                    ds[var_name].attrs['axis'] = 'Z'

                try:
                    ds[var_name].attrs['positive']
                except KeyError:
                    ds[var_name].attrs['positive'] = 'up'

            # Check if the Conventions global attribute lists the CF convention
            try:
                Conventions = ds.attrs['Conventions']
                Conventions = Conventions.split()
                cf_listed = False
                for ii in Conventions:
                    if ii.startswith('CF-'):
                        cf_listed = True
                        break
                if not cf_listed:
                    Conventions.append(cf_convention)
                ds.attrs['Conventions'] = ' '.join(Conventions)

            except KeyError:
                ds.attrs['Conventions'] = str(cf_convention)

            # Reorder global attributes to ensure history is last
            try:
                history = copy.copy(ds.attrs['history'])
                del ds.attrs['history']
                ds.attrs['history'] = history
            except KeyError:
                pass

        current_time = dt.datetime.utcnow().replace(microsecond=0)
        history_value = (
            f'Written to file by ACT-{act.__version__} '
            f'with write_netcdf() at {current_time} UTC'
        )
        if 'history' in list(ds.attrs.keys()):
            ds.attrs['history'] += f" ; {history_value}"
        else:
            ds.attrs['history'] = history_value

        ds.to_netcdf(encoding=encoding, **kwargs)


def check_if_tar_gz_file(filenames):
    """
    Unpacks gunzip and/or TAR file contents and returns Xarray Dataset

    ...

    Parameters
    ----------
    filenames : str, pathlib.Path
        Filenames to check if gunzip and/or tar files.


    Returns
    -------
    filenames : Paths to extracted files from gunzip or TAR files

    """

    cleanup = False
    if isinstance(filenames, (str, PathLike)):
        try:
            if is_gunzip_file(filenames) or tarfile.is_tarfile(str(filenames)):
                tmpdirname = tempfile.mkdtemp()
                cleanup = True
                if is_gunzip_file(filenames):
                    filenames = unpack_gzip(filenames, write_directory=tmpdirname)

                if tarfile.is_tarfile(str(filenames)):
                    filenames = unpack_tar(filenames, write_directory=tmpdirname, randomize=False)
        except Exception:
            pass

    return filenames, cleanup


def read_arm_mmcr(filenames):
    """

    Reads in ARM MMCR files and splits up the variables into specific
    mode variables based on what's in the files.  MMCR files have the modes
    interleaved and are not readable using xarray so some modifications are
    needed ahead of time.

    Parameters
    ----------
    filenames : str, pathlib.PosixPath or list of str
        Name of file(s) to read.

    Returns
    -------
    ds : xarray.Dataset (or None)
        ACT Xarray dataset (or None if no data file(s) found).

    """

    # Sort the files to make sure they concatenate right
    filenames.sort()

    # Run through each file and read it in using netCDF4, then
    # read it in with xarray
    multi_ds = []
    for f in filenames:
        nc = Dataset(f, 'a')
        # Change heights name to range to read appropriately to xarray
        if 'heights' in nc.dimensions:
            nc.renameDimension('heights', 'range')
        if nc is not None:
            ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))
            multi_ds.append(ds)
    # Concatenate datasets together
    if len(multi_ds) > 1:
        ds = xr.concat(multi_ds, dim='time')
    else:
        ds = multi_ds[0]

    # Get mdoes and ranges with time/height modes
    modes = ds['mode'].values
    mode_vars = []
    for v in ds:
        if 'range' in ds[v].dims and 'time' in ds[v].dims and len(ds[v].dims) == 2:
            mode_vars.append(v)

    # For each mode, run extract data variables if available
    # saves as individual variables in the file.
    for m in modes:
        if len(ds['ModeDescription'].shape) > 1:
            mode_desc = ds['ModeDescription'].values[0, m]
            if np.isnan(ds['heights'].values[0, m, :]).all():
                continue
            range_data = ds['heights'].values[0, m, :]
        else:
            mode_desc = ds['ModeDescription'].values[m]
            if np.isnan(ds['heights'].values[m, :]).all():
                continue
            range_data = ds['heights'].values[m, :]
        mode_desc = str(mode_desc).split('_')[-1][0:-1]
        mode_desc = str(mode_desc).split('\'')[0]
        idx = np.where(ds['ModeNum'].values == m)[0]
        idy = np.where(~np.isnan(range_data))[0]
        for v in mode_vars:
            new_var_name = v + '_' + mode_desc
            time_name = 'time_' + mode_desc
            range_name = 'range_' + mode_desc
            data = ds[v].values[idx, :]
            data = data[:, idy]
            attrs = ds[v].attrs
            da = xr.DataArray(
                data=data,
                coords={time_name: ds['time'].values[idx], range_name: range_data[idy]},
                dims=[time_name, range_name],
                attrs=attrs,
            )
            ds[new_var_name] = da

    return ds
