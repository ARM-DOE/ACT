"""
Module containing utilities for the data.

"""

import importlib
import warnings

import json
import metpy
import numpy as np
import pint
import scipy.stats as stats
import xarray as xr
from pathlib import Path
import re
import requests

spec = importlib.util.find_spec('pyart')
if spec is not None:
    PYART_AVAILABLE = True
else:
    PYART_AVAILABLE = False


@xr.register_dataset_accessor('utils')
class ChangeUnits:
    """
    Class for updating units in the dataset. Data values and units attribute
    are updated in place. Coordinate variables can not be updated in place. Must
    use new returned dataset when updating coordinage varibles.
    """

    def __init__(self, ds):
        self._ds = ds

    def change_units(
        self,
        variables=None,
        desired_unit=None,
        skip_variables=None,
        skip_standard=True,
        verbose=False,
        raise_error=False,
    ):
        """
        Parameters
        ----------
        variables : None, str or list of str
            Variable names to attempt to change units.
        desired_unit : str
            Desired udunits unit string.
        skip_variables : None, str or list of str
            Variable names to skip. Works well when not providing a variables
            keyword.
        skip_standard : boolean
            Flag indicating the QC variables that will not need changing are
            skipped. Makes the processing faster when processing all variables
            in dataset.
        verbose : boolean
            Option to print statement when an attempted conversion fails. Set to False
            as default because many units strings are not udunits complient and when
            trying to convert all varialbes of a type of units (eg temperature) the code
            can print a lot of unecessary information.
        raise_error : boolean
            Raise an error if conversion is not successful.

        Returns
        -------
        dataset : xarray.dataset
            A new dataset if the coordinate variables are updated. Required to
            use returned dataset if coordinage variabels are updated,
            otherwise the dataset is updated in place.
        """

        if variables is not None and isinstance(variables, str):
            variables = [variables]

        if skip_variables is not None and isinstance(skip_variables, str):
            skip_variables = [skip_variables]

        if desired_unit is None:
            raise ValueError("Need to provide 'desired_unit' keyword for .change_units() method")

        if variables is None:
            variables = list(self._ds.data_vars)

        if skip_variables is not None:
            variables = list(set(variables) - set(skip_variables))

        for var_name in variables:
            try:
                if self._ds[var_name].attrs['standard_name'] == 'quality_flag':
                    continue
            except KeyError:
                pass

            try:
                data = convert_units(
                    self._ds[var_name].values,
                    self._ds[var_name].attrs['units'],
                    desired_unit,
                )
                try:
                    self._ds[var_name].values = data
                    self._ds[var_name].attrs['units'] = desired_unit
                except ValueError:
                    attrs = self._ds[var_name].attrs
                    self._ds = self._ds.assign_coords({var_name: data})
                    attrs['units'] = desired_unit
                    self._ds[var_name].attrs = attrs
            except (
                KeyError,
                pint.errors.DimensionalityError,
                pint.errors.UndefinedUnitError,
                np.core._exceptions.UFuncTypeError,
            ):
                if raise_error:
                    raise ValueError(
                        f"Unable to convert '{var_name}' to units of '{desired_unit}'."
                    )
                elif verbose:
                    print(
                        f"\n    Unable to convert '{var_name}' to units of '{desired_unit}'. "
                        f"Skipping unit converstion for '{var_name}'.\n"
                    )

        return self._ds


# @xr.register_dataset_accessor('utils')
class DatastreamParserARM:
    '''
    Class to parse ARM datastream names or filenames into its components.
    Will return None for each attribute if not extracted from the filename.

    Attributes
    ----------
    site : str or None
        The site code extracted from the filename.
    datastream_class : str
        The datastream class extracted from the filename.
    facility : str or None
        The datastream facility code extracted from the filename.
    level : str or None
        The datastream level code extracted from the filename.
    datastream : str or None
        The datastram extracted from the filename.
    date : str or None
        The date extracted from the filename.
    time : str or None
        The time extracted from the filename.
    ext : str or None
        The file extension extracted from the filename.

    Example
    -------
    >>> from act.utils.data_utils import DatastreamParserARM
    >>> file = 'sgpmetE13.b1.20190501.024254.nc'
    >>> fn_obj = DatastreamParserARM(file)
    >>> fn_obj.site
    'sgp'
    >>> fn_obj.datastream_class
    'met'


    '''

    def __init__(self, ds=''):
        '''
        Constructor that initializes datastream data member and runs
        parse_datastream class method.  Also converts datastream name to
        lower case before parsing.

        ds : str
            The datastream or filename to parse

        '''

        if isinstance(ds, str):
            self.__datastream = Path(ds).name
        else:
            raise ValueError('Datastream or filename name must be a string')

        try:
            self.__parse_datastream()
        except ValueError:
            self.__site = None
            self.__class = None
            self.__facility = None
            self.__datastream = None
            self.__level = None
            self.__date = None
            self.__time = None
            self.__ext = None

    def __parse_datastream(self):
        '''
        Private method to parse datastream name into its various components
        (site, class, facility, and data level.  Is called automatically by
        constructor when object of class is instantiated and when the
        set_datastream method is called to reset the object.

        '''
        # Import the built-in match function from regular expression library
        # self.__datastream = self.__datastream
        tempstring = self.__datastream.split('.')

        # Check to see if ARM-standard filename was passed
        self.__ext = None
        self.__time = None
        self.__date = None
        self.__level = None
        self.__site = None
        self.__class = None
        self.__facility = None
        if len(tempstring) >= 5:
            self.__ext = tempstring[4]

        if len(tempstring) >= 4:
            self.__time = tempstring[3]

        if len(tempstring) >= 3:
            self.__date = tempstring[2]

        if len(tempstring) >= 2:
            m = re.match('[abcs0][0123456789]', tempstring[1])
            if m is not None:
                self.__level = m.group()

        match = False
        m = re.search(r'(^[a-z]{3})(\w+)([A-Z]{1}\d{1,2})$', tempstring[0])
        if m is not None:
            self.__site = m.group(1)
            self.__class = m.group(2)
            self.__facility = m.group(3)
            match = True

        if not match:
            m = re.search(r'(^[a-z]{3})(\w+)$', tempstring[0])
            if m is not None:
                self.__site = m.group(1)
                self.__class = m.group(2)
                match = True

        if not match and len(tempstring[0]) == 3:
            self.__site = tempstring[0]
            match = True

        if not match:
            raise ValueError(self.__datastream)

    def set_datastream(self, ds):
        '''
        Method used to set or reset object by passing a new datastream name.

        '''

        self.__init__(ds)

    @property
    def datastream(self):
        '''
        Property returning current datastream name stored in object in
        standard lower case.  Will return the datastrem with no level if
        unavailable.

        '''

        try:
            return ''.join((self.__site, self.__class, self.__facility, '.', self.__level))
        except TypeError:
            return None

    @property
    def site(self):
        '''
        Property returning current site name stored in object in standard
        lower case.

        '''

        return self.__site

    @property
    def datastream_class(self):
        '''
        Property returning current datastream class name stored in object in
        standard lower case.  Could not use class as attribute name since it
        is a reserved word in Python

        '''

        return self.__class

    @property
    def facility(self):
        '''
        Property returning current facility name stored in object in
        standard upper case.

        '''

        try:
            return self.__facility.upper()
        except AttributeError:
            return self.__facility

    @property
    def level(self):
        '''
        Property returning current data level stored in object in standard
        lower case.
        '''

        return self.__level

    @property
    def datastream_standard(self):
        '''
        Property returning datastream name in ARM-standard format with
        facility in caps.  Will return the datastream name with no level if
        unavailable.
        '''

        try:
            return ''.join((self.site, self.datastream_class, self.facility, '.', self.level))

        except TypeError:
            return None

    @property
    def date(self):
        '''
        Property returning date from filename.
        '''

        return self.__date

    @property
    def time(self):
        '''
        Property returning time from filename.
        '''

        return self.__time

    @property
    def ext(self):
        '''
        Property returning file extension from filename.
        '''

        return self.__ext


def assign_coordinates(ds, coord_list):
    """
    This procedure will create a new ACT dataset whose coordinates are
    designated to be the variables in a given list. This helps make data
    slicing via xarray and visualization easier.

    Parameters
    ----------
    ds : ACT Dataset
        The ACT Dataset to modify the coordinates of.
    coord_list : dict
        The list of variables to assign as coordinates, given as a dictionary
        whose keys are the variable name and values are the dimension name.

    Returns
    -------
    new_ds : ACT Dataset
        The new ACT Dataset with the coordinates assigned to be the given
        variables.

    """
    # Check to make sure that user assigned valid entries for coordinates

    for coord in coord_list.keys():
        if coord not in ds.variables.keys():
            raise KeyError(coord + ' is not a variable in the Dataset.')

        if ds.dims[coord_list[coord]] != len(ds.variables[coord]):
            raise IndexError(
                coord + ' must have the same ' + 'value as length of ' + coord_list[coord]
            )

    new_ds_dict = {}
    for variable in ds.variables.keys():
        my_coord_dict = {}
        dataarray = ds[variable]
        if len(dataarray.dims) > 0:
            for coord in coord_list.keys():
                if coord_list[coord] in dataarray.dims:
                    my_coord_dict[coord_list[coord]] = ds[coord]

        if variable not in my_coord_dict.keys() and variable not in ds.dims:
            the_dataarray = xr.DataArray(dataarray.data, coords=my_coord_dict, dims=dataarray.dims)
            new_ds_dict[variable] = the_dataarray

    new_ds = xr.Dataset(new_ds_dict, coords=my_coord_dict)

    return new_ds


def add_in_nan(time, data):
    """
    This procedure adds in NaNs when there is a larger than expected time step.
    This is useful for timeseries where there is a gap in data and need a
    NaN value to stop plotting from connecting data over the large data gap.

    Parameters
    ----------
    time : 1D array of numpy datetime64 or Xarray DataArray of datetime64
        Times in the timeseries.
    data : 1D or 2D numpy array or Xarray DataArray
        Array containing the data. The 0 axis corresponds to time.

    Returns
    -------
    time : numpy array or Xarray DataArray
        The array containing the new times including a NaN filled
        sampe or slice if multi-dimensional.
        The intervals are determined by the mode of the timestep in *time*.
    data : numpy array or Xarray DataArray
        The array containing the NaN-indserted data.

    """

    time_is_DataArray = False
    data_is_DataArray = False
    if isinstance(time, xr.core.dataarray.DataArray):
        time_is_DataArray = True
        time_attributes = time.attrs
        time_dims = time.dims
    if isinstance(data, xr.core.dataarray.DataArray):
        data_is_DataArray = True
        data_attributes = data.attrs
        data_dims = data.dims

    # Return if time dimension is only size one since we can't do differences.
    if time.size > 2:
        data = np.asarray(data)
        time = np.asarray(time)
        # Not sure if we need to set to second data type to make it work better.
        # Leaving code in here in case we need to update.
        # diff = np.diff(time.astype('datetime64[s]'), 1)
        diff = np.diff(time, 1)

        # Wrapping in a try to catch error while switching between numpy 1.10 to 1.11
        try:
            mode = stats.mode(diff, keepdims=True).mode[0]
        except TypeError:
            mode = stats.mode(diff).mode[0]
        index = np.where(diff > (2.0 * mode))

        offset = 0
        for i in index[0]:
            corr_i = i + offset

            if len(data.shape) == 1:
                # For line plotting adding a NaN will stop the connection of the line
                # between points. So we just need to add a NaN anywhere between the points.
                corr_i = i + offset
                time_added = time[corr_i] + (time[corr_i + 1] - time[corr_i]) / 2.0
                time = np.insert(time, corr_i + 1, time_added)
                data = np.insert(data, corr_i + 1, np.nan, axis=0)
                offset += 1
            else:
                # For 2D plots need to add a NaN right after and right before the data
                # to correctly mitigate streaking with pcolormesh.
                time_added_1 = time[corr_i] + 1  # One time step after
                time_added_2 = time[corr_i + 1] - 1  # One time step before
                time = np.insert(time, corr_i + 1, [time_added_1, time_added_2])
                data = np.insert(data, corr_i + 1, np.nan, axis=0)
                data = np.insert(data, corr_i + 2, np.nan, axis=0)
                offset += 2

        if time_is_DataArray:
            time = xr.DataArray(time, attrs=time_attributes, dims=time_dims)

        if data_is_DataArray:
            data = xr.DataArray(data, attrs=data_attributes, dims=data_dims)

    return time, data


def get_missing_value(
    ds,
    variable,
    default=-9999,
    add_if_missing_in_ds=False,
    use_FillValue=False,
    nodefault=False,
):
    """
    Function to get missing value from missing_value or _FillValue attribute.
    Works well with catching errors and allows for a default value when a
    missing value is not listed in the dataset. You may get strange results
    becaus xarray will automatically convert all missing_value or
    _FillValue to NaN and then remove the missing_value and
    _FillValue variable attribute when reading data with default settings.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset containing data variable.
    variable : str
        Variable name to use for getting missing value.
    default : int or float
        Default value to use if missing value attribute is not in dataset.
    add_if_missing_in_ds : bool
        Boolean to add to the dataset if does not exist. Default is False.
    use_FillValue : bool
        Boolean to use _FillValue instead of missing_value. If missing_value
        does exist and _FillValue does not will add _FillValue
        set to missing_value value.
    nodefault : bool
        Option to use this to check if the varible has a missing value set and
        do not want to get default as return. If the missing value is found
        will return, else will return None.

    Returns
    -------
    missing : scalar int or float (or None)
        Value used to indicate missing value matching type of data or None if
        nodefault keyword set to True.

    Examples
    --------
    .. code-block:: python

        from act.utils import get_missing_value

        missing = get_missing_value(dq_ds, "temp_mean")
        print(missing)
        -9999.0

    """
    in_ds = False
    if use_FillValue:
        missing_atts = ['_FillValue', 'missing_value']
    else:
        missing_atts = ['missing_value', '_FillValue']

    for att in missing_atts:
        try:
            missing = ds[variable].attrs[att]
            in_ds = True
            break
        except (AttributeError, KeyError):
            missing = default

    # Check if do not want a default value retured and a value
    # was not fund.
    if nodefault is True and in_ds is False:
        missing = None
        return missing

    # Check data type and try to match missing_value to the data type of data
    try:
        missing = ds[variable].data.dtype.type(missing)
    except KeyError:
        pass
    except AttributeError:
        print(
            ('--- AttributeError: Issue trying to get data type ' + 'from "{}" data ---').format(
                variable
            )
        )

    # If requested add missing value to the dataset
    if add_if_missing_in_ds and not in_ds:
        try:
            ds[variable].attrs[missing_atts[0]] = missing
        except KeyError:
            print(
                ('---  KeyError: Issue trying to add "{}" ' + 'attribute to "{}" ---').format(
                    missing_atts[0], variable
                )
            )

    return missing


def convert_units(data, in_units, out_units):
    """
    Wrapper function around library to convert data using unit strings.
    Currently using pint units library. Will attempt to preserve numpy
    data type, but will upconvert to numpy float64 if need to change
    data type for converted values.

    Parameters
    ----------
    data : list, tuple or numpy array
        Data array to be modified.
    in_units : str
        Units scalar string of input data array.
    out_units : str
        Units scalar string of desired output data array.

    Returns
    -------
    data : numpy array
        Data array converted into new units.

    Examples
    --------
    > data = np.array([1,2,3,4,5,6])
    > data = convert_units(data, 'cm', 'm')
    > data
    array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])


    """
    # Fix historical and current incorrect usage of units.
    convert_dict = {
        'C': 'degC',
        'F': 'degF',
        '%': 'percent',  # Pint does not like this symbol with .to('%')
        '1': 'unitless',  # Pint does not like a number
    }

    if in_units in convert_dict:
        in_units = convert_dict[in_units]

    if out_units in convert_dict:
        out_units = convert_dict[out_units]

    if in_units == out_units:
        return data

    # Instantiate the registry
    ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)

    # Add missing units and conversions
    ureg.define('fraction = []')
    ureg.define('unitless = []')

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    data_type = data.dtype
    data_type_kind = data.dtype.kind

    # Do the conversion magic
    data = (data * ureg(in_units)).to(out_units)
    data = data.magnitude

    # The data type may be changed by pint. This is a side effect
    # of pint changing the datatype to float. Check if the converted values
    # need float precision. If so leave, if not change back to orginal
    # precision after checking if the precsion is not lost with the orginal
    # data type.
    if (
        data_type_kind == 'i'
        and np.nanmin(data) >= np.iinfo(data_type).min
        and np.nanmax(data) <= np.iinfo(data_type).max
        and np.all(np.mod(data, 1) == 0)
    ):
        data = data.astype(data_type)

    return data


def ts_weighted_average(ts_dict):
    """
    Program to take in multiple difference time-series and average them
    using the weights provided. This assumes that the variables passed in
    all have the same units. Please see example gallery for an example.

    NOTE: All weights should add up to 1

    Parameters
    ----------
    ts_dict : dict
        Dictionary containing datastream, variable, weight, and datasets

        .. code-block:: python

            t_dict = {
                "sgpvdisC1.b1": {
                    "variable": "rain_rate",
                    "weight": 0.05,
                    "ds": ds,
                },
                "sgpmetE13.b1": {
                    "variable": [
                        "tbrg_precip_total",
                        "org_precip_rate_mean",
                        "pwd_precip_rate_mean_1min",
                    ],
                    "weight": [0.25, 0.05, 0.0125],
                },
            }

    Returns
    -------
    data : numpy array
        Variable of time-series averaged data

    """

    # Run through each datastream/variable and get data
    da_array = []
    data = 0.0
    for d in ts_dict:
        for i, v in enumerate(ts_dict[d]['variable']):
            new_name = '_'.join([d, v])
            # Since many variables may have same name, rename with datastream
            da = ts_dict[d]['ds'][v].rename(new_name)

            # Apply Weights to Data
            da.values = da.values * ts_dict[d]['weight'][i]
            da_array.append(da)

    da = xr.merge(da_array)

    # Stack all the data into a 2D time series
    data = None
    for i, d in enumerate(da):
        if i == 0:
            data = da[d].values
        else:
            data = np.vstack((data, da[d].values))

    # Sum data across each time sample
    data = np.nansum(data, 0)

    # Add data to data array and return
    dims = ts_dict[list(ts_dict.keys())[0]]['ds'].dims
    da_xr = xr.DataArray(
        data,
        dims=dims,
        coords={'time': ts_dict[list(ts_dict.keys())[0]]['ds']['time']},
    )
    da_xr.attrs['long_name'] = 'Weighted average of ' + ', '.join(list(ts_dict.keys()))

    return da_xr


def accumulate_precip(ds, variable, time_delta=None):
    """
    Program to accumulate rain rates from an act xarray dataset and insert
    variable back into an act xarray dataset with "_accumulated" appended to
    the variable name. Please verify that your units are accurately described
    in the data.

    Parameters
    ----------
    ds : xarray.DataSet
        ACT Xarray dataset.
    variable : string
        Variable name.
    time_delta : float
        Time delta to caculate precip accumulations over.
        Useful if full time series is not passed in.

    Returns
    -------
    ds : xarray.DataSet
        ACT Xarray dataset with variable_accumulated.

    """
    # Get Data, time, and metadat
    data = ds[variable]
    time = ds.coords['time']
    units = ds[variable].attrs['units']

    # Calculate mode of the time samples(i.e. 1 min vs 1 sec)
    if time_delta is None:
        diff = np.diff(time.values, 1) / np.timedelta64(1, 's')
        try:
            t_delta = stats.mode(diff, keepdims=False).mode
        except TypeError:
            t_delta = stats.mode(diff).mode
    else:
        t_delta = time_delta

    # Calculate the accumulation based on the units
    t_factor = t_delta / 60.0
    if units == 'mm/hr':
        data = data * (t_factor / 60.0)

    accum = np.nancumsum(data.values)

    # Add accumulated variable back to the dataset
    long_name = 'Accumulated precipitation'
    attrs = {'long_name': long_name, 'units': 'mm'}
    ds['_'.join([variable, 'accumulated'])] = xr.DataArray(
        accum, coords=ds[variable].coords, attrs=attrs
    )

    return ds


def create_pyart_obj(
    ds,
    variables=None,
    sweep=None,
    azimuth=None,
    elevation=None,
    range_var=None,
    sweep_start=None,
    sweep_end=None,
    lat=None,
    lon=None,
    alt=None,
    sweep_mode='ppi',
    sweep_az_thresh=10.0,
    sweep_el_thresh=0.5,
):
    """
    Produces a Py-ART radar object based on data in the ACT Xarray dataset.

    Parameters
    ----------
    ds : xarray.DataSet
        ACT Xarray dataset.
    variables : list
        List of variables to add to the radar object, will default to all
        variables.
    sweep : string
        Name of variable that has sweep information. If none, will try and
        calculate from the azimuth and elevation.
    azimuth : string
        Name of azimuth variable. Will try and find one if none given.
    elevation : string
        Name of elevation variable. Will try and find one if none given.
    range_var : string
        Name of the range variable. Will try and find one if none given.
    sweep_start : string
        Name of variable with sweep start indices.
    sweep_end : string
        Name of variable with sweep end indices.
    lat : string
        Name of latitude variable. Will try and find one if none given.
    lon : string
        Name of longitude variable. Will try and find one if none given.
    alt : string
        Name of altitude variable. Will try and find one if none given.
    sweep_mode : string
        Type of scan. Defaults to PPI.
    sweep_az_thresh : float
        If calculating sweep numbers, the maximum change in azimuth before new
        sweep.
    sweep_el_thresh : float
        If calculating sweep numbers, the maximum change in elevation before
        new sweep.

    Returns
    -------
    radar : radar.Radar
        Py-ART Radar Object.

    """

    if not PYART_AVAILABLE:
        raise ImportError(
            'Py-ART needs to be installed on your system to convert to ' 'Py-ART Object.'
        )
    else:
        import pyart
    # Get list of variables if none provided
    if variables is None:
        variables = list(ds.keys())

    # Determine the sweeps if not already in a variable$a
    if sweep is None:
        swp = np.zeros(ds.sizes['time'])
        for key in ds.variables.keys():
            if len(ds.variables[key].shape) == 2:
                total_rays = ds.variables[key].shape[0]
                break
        nsweeps = int(total_rays / ds.variables['time'].shape[0])
    else:
        swp = ds[sweep].values
        nsweeps = ds[sweep].values

    # Get coordinate variables
    if lat is None:
        lat = [s for s in variables if 'latitude' in s]
        if len(lat) == 0:
            lat = [s for s in variables if 'lat' in s]
        if len(lat) == 0:
            raise ValueError(
                'Latitude variable not set and could not be ' 'discerned from the data.'
            )
        else:
            lat = lat[0]

    if lon is None:
        lon = [s for s in variables if 'longitude' in s]
        if len(lon) == 0:
            lon = [s for s in variables if 'lon' in s]
        if len(lon) == 0:
            raise ValueError(
                'Longitude variable not set and could not be ' 'discerned from the data.'
            )
        else:
            lon = lon[0]

    if alt is None:
        alt = [s for s in variables if 'altitude' in s]
        if len(alt) == 0:
            alt = [s for s in variables if 'alt' in s]
        if len(alt) == 0:
            raise ValueError(
                'Altitude variable not set and could not be ' 'discerned from the data.'
            )
        else:
            alt = alt[0]

    # Get additional variable names if none provided
    if azimuth is None:
        azimuth = [s for s in sorted(variables) if 'azimuth' in s][0]
        if len(azimuth) == 0:
            raise ValueError(
                'Azimuth variable not set and could not be ' 'discerned from the data.'
            )

    if elevation is None:
        elevation = [s for s in sorted(variables) if 'elevation' in s][0]
        if len(elevation) == 0:
            raise ValueError(
                'Elevation variable not set and could not be ' 'discerned from the data.'
            )

    if range_var is None:
        range_var = [s for s in sorted(variables) if 'range' in s][0]
        if len(range_var) == 0:
            raise ValueError('Range variable not set and could not be ' 'discerned from the data.')

    # Calculate the sweep indices if not passed in
    if sweep_start is None and sweep_end is None:
        az_diff = np.abs(np.diff(ds[azimuth].values))
        az_idx = az_diff > sweep_az_thresh

        el_diff = np.abs(np.diff(ds[elevation].values))
        el_idx = el_diff > sweep_el_thresh

        # Create index list
        az_index = list(np.where(az_idx)[0] + 1)
        el_index = list(np.where(el_idx)[0] + 1)
        index = sorted(az_index + el_index)

        index.insert(0, 0)
        index += [ds.sizes['time']]

        sweep_start_index = []
        sweep_end_index = []
        for i in range(len(index) - 1):
            sweep_start_index.append(index[i])
            sweep_end_index.append(index[i + 1] - 1)
            swp[index[i] : index[i + 1]] = i
    else:
        sweep_start_index = ds[sweep_start].values
        sweep_end_index = ds[sweep_end].values
        if sweep is None:
            for i in range(len(sweep_start_index)):
                swp[sweep_start_index[i] : sweep_end_index[i]] = i

    radar = pyart.testing.make_empty_ppi_radar(ds.sizes[range_var], ds.sizes['time'], nsweeps)

    radar.time['data'] = np.array(ds['time'].values)

    # Add lat, lon, and alt
    radar.latitude['data'] = np.array(ds[lat].values)
    radar.longitude['data'] = np.array(ds[lon].values)
    radar.altitude['data'] = np.array(ds[alt].values)

    # Add sweep information
    radar.sweep_number['data'] = swp
    radar.sweep_start_ray_index['data'] = sweep_start_index
    radar.sweep_end_ray_index['data'] = sweep_end_index
    radar.sweep_mode['data'] = np.array(sweep_mode)
    radar.scan_type = sweep_mode

    # Add elevation, azimuth, etc...
    radar.azimuth['data'] = np.array(ds[azimuth])
    radar.elevation['data'] = np.array(ds[elevation])
    radar.fixed_angle['data'] = np.array(ds[elevation].values[0])
    radar.range['data'] = np.array(ds[range_var].values)

    # Calculate radar points in lat/lon
    radar.init_gate_altitude()
    radar.init_gate_longitude_latitude()

    # Add the fields to the radar object
    fields = {}
    for v in variables:
        ref_dict = pyart.config.get_metadata(v)
        ref_dict['data'] = np.array(ds[v].values)
        fields[v] = ref_dict
    radar.fields = fields

    return radar


def convert_to_potential_temp(
    ds=None,
    temp_var_name=None,
    press_var_name=None,
    temperature=None,
    pressure=None,
    temp_var_units=None,
    press_var_units=None,
):
    """
    Converts temperature to potential temperature.

    Parameters
    ----------
    ds : xarray.DataSet
        ACT Xarray dataset
    temp_var_name : str
        Temperature variable name in the ACT Xarray dataset containing
        temperature data to convert.
    press_var_name : str
        Pressure variable name in the ACT Xarray dataset containing the
        pressure data to use in conversion. If not set or set to None will
        use values from pressure keyword.
    pressure : int, float, numpy array
        Optional pressure values to use instead of using values from xarray
        dataset. If set must also set press_var_units keyword.
    temp_var_units : string
        Pint recognized units string for temperature data. If set to None will
        use the units attribute under temperature variable in ds.
    press_var_units : string
        Pint recognized units string for pressure data. If set to None will
        use the units attribute under pressure variable in the dataset. If using
        the pressure keyword this must be set.

    Returns
    -------
    potential_temperature : None, int, float, numpy array
        The converted temperature to potential temperature or None if something
        goes wrong.

    References
    ----------
    May, R. M., Arms, S. C., Marsh, P., Bruning, E., Leeman, J. R., Goebbert,
    K., Thielen, J. E., and Bruick, Z., 2021: MetPy: A Python Package for
    Meteorological Data. Unidata, https://github.com/Unidata/MetPy,
    doi:10.5065/D6WW7G29.

    """
    potential_temp = None
    if temp_var_units is None and temp_var_name is not None:
        temp_var_units = ds[temp_var_name].attrs['units']
    if press_var_units is None and press_var_name is not None:
        press_var_units = ds[press_var_name].attrs['units']

    if press_var_units is None:
        raise ValueError(
            "Need to provide 'press_var_units' keyword " "when using 'pressure' keyword"
        )
    if temp_var_units is None:
        raise ValueError(
            "Need to provide 'temp_var_units' keyword " "when using 'temperature' keyword"
        )

    if temperature is not None:
        temperature = metpy.units.units.Quantity(temperature, temp_var_units)
    else:
        temperature = metpy.units.units.Quantity(ds[temp_var_name].values, temp_var_units)

    if pressure is not None:
        pressure = metpy.units.units.Quantity(pressure, press_var_units)
    else:
        pressure = metpy.units.units.Quantity(ds[press_var_name].values, press_var_units)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        potential_temp = metpy.calc.potential_temperature(pressure, temperature)
    potential_temp = potential_temp.to(temp_var_units).magnitude

    return potential_temp


def height_adjusted_temperature(
    ds=None,
    temp_var_name=None,
    height_difference=0,
    height_units='m',
    press_var_name=None,
    temperature=None,
    temp_var_units=None,
    pressure=101.325,
    press_var_units='kPa',
):
    """
    Converts temperature for change in height.

    Parameters
    ----------
    ds : xarray.DataSet, None
        Optional Xarray dataset for retrieving pressure and temperature values.
        Not needed if using temperature keyword.
    temp_var_name : str, None
        Optional temperature variable name in the Xarray dataset containing the
        temperature data to use in conversion. If not set or set to None will
        use values from temperature keyword.
    height_difference : int, float
        Required difference in height to adjust pressure values. Positive
        values to increase height negative values to decrease height.
    height_units : str
        Units of height value.
    press_var_name : str, None
        Optional pressure variable name in the Xarray dataset containing the
        pressure data to use in conversion. If not set or set to None will
        use values from pressure keyword.
    temperature : int, float, numpy array, None
        Optional temperature values to use instead of values in the dataset.
    temp_var_units : str, None
        Pint recognized units string for temperature data. If set to None will
        use the units attribute under temperature variable in the dataset.
        If using the temperature keyword this must be set.
    pressure : int, float, numpy array, None
        Optional pressure values to use instead of values in the dataset.
        Default value of sea level pressure is set for ease of use.
    press_var_units : str, None
        Pint recognized units string for pressure data. If set to None will
        use the units attribute under pressure variable in the dataset.
        If using the pressure keyword this must be set. Default value of
        sea level pressure is set for ease of use.

    Returns
    -------
    adjusted_temperature : None, int, float, numpy array
        The height adjusted temperature or None if something goes wrong.

    References
    ----------
    May, R. M., Arms, S. C., Marsh, P., Bruning, E., Leeman, J. R., Goebbert,
    K., Thielen, J. E., and Bruick, Z., 2021: MetPy: A Python Package for
    Meteorological Data. Unidata, https://github.com/Unidata/MetPy,
    doi:10.5065/D6WW7G29.

    """
    adjusted_temperature = None
    if temp_var_units is None and temperature is None:
        temp_var_units = ds[temp_var_name].attrs['units']
    if temp_var_units is None:
        raise ValueError(
            "Need to provide 'temp_var_units' keyword when " 'providing temperature keyword values.'
        )

    if temperature is not None:
        temperature = metpy.units.units.Quantity(temperature, temp_var_units)
    else:
        temperature = metpy.units.units.Quantity(ds[temp_var_name].values, temp_var_units)

    if press_var_name is not None:
        pressure = metpy.units.units.Quantity(ds[press_var_name].values, press_var_units)
    else:
        pressure = metpy.units.units.Quantity(pressure, press_var_units)

    adjusted_pressure = height_adjusted_pressure(
        height_difference=height_difference,
        height_units=height_units,
        pressure=pressure.magnitude,
        press_var_units=press_var_units,
    )
    adjusted_pressure = metpy.units.units.Quantity(adjusted_pressure, press_var_units)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        adjusted_temperature = metpy.calc.dry_lapse(adjusted_pressure, temperature, pressure)
    adjusted_temperature = adjusted_temperature.to(temp_var_units).magnitude

    return adjusted_temperature


def height_adjusted_pressure(
    ds=None,
    press_var_name=None,
    height_difference=0,
    height_units='m',
    pressure=None,
    press_var_units=None,
):
    """
    Converts pressure for change in height.

    Parameters
    ----------
    ds : xarray.DataSet, None
        Optional Xarray dataset for retrieving pressure values. Not needed if
        using pressure keyword.
    press_var_name : str, None
        Optional pressure variable name in the Xarray dataset containing the
        pressure data to use in conversion. If not set or set to None will
        use values from pressure keyword.
    height_difference : int, float
        Required difference in height to adjust pressure values. Positive
        values to increase height negative values to decrease height.
    height_units : str
        Units of height value.
    pressure : int, float, numpy array, None
        Optional pressure values to use instead of values in the dataset.
    press_var_units : str, None
        Pint recognized units string for pressure data. If set to None will
        use the units attribute under pressure variable in the dataset.
        If using the pressure keyword this must be set.

    Returns
    -------
    adjusted_pressure : None, int, float, numpy array
        The height adjusted pressure or None if something goes wrong.

    References
    ----------
    May, R. M., Arms, S. C., Marsh, P., Bruning, E., Leeman, J. R., Goebbert,
    K., Thielen, J. E., and Bruick, Z., 2021: MetPy: A Python Package for
    Meteorological Data. Unidata, https://github.com/Unidata/MetPy,
    doi:10.5065/D6WW7G29.

    """
    adjusted_pressure = None
    if press_var_units is None and pressure is None:
        press_var_units = ds[press_var_name].attrs['units']

    if press_var_units is None:
        raise ValueError(
            "Need to provide 'press_var_units' keyword when " 'providing pressure keyword values.'
        )

    if pressure is not None:
        pressure = metpy.units.units.Quantity(pressure, press_var_units)
    else:
        pressure = metpy.units.units.Quantity(ds[press_var_name].values, press_var_units)

    height_difference = metpy.units.units.Quantity(height_difference, height_units)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        adjusted_pressure = metpy.calc.add_height_to_pressure(pressure, height_difference)
    adjusted_pressure = adjusted_pressure.to(press_var_units).magnitude

    return adjusted_pressure


def arm_site_location_search(site_code='sgp', facility_code=None):
    """
    Parameters
    ----------
    site_code : str
        ARM site code to retrieve facilities and coordinate information. Example and default
        is 'sgp'.
    facility_code : str or None
        Facility code or codes for the ARM site provided. If None is provided, all facilities are returned.
        Example string for multiple facilities is 'A4,I5'.

    Returns
    -------
    coord_dict : dict
        A dictionary containing the facility chosen coordinate information or all facilities
        if None for facility_code and their respective coordinates.

    """
    headers = {
        'Content-Type': 'application/json',
    }
    # Return all facilities if facility_code is None else set the query to include
    # facility search
    if facility_code is None:
        query = "site_code:" + site_code
    else:
        query = "site_code:" + site_code + " AND facility_code:" + facility_code

    # Search aggregation for elastic search
    json_data = {
        "aggs": {
            "distinct_facility_code": {
                "terms": {
                    "field": "facility_code.keyword",
                    "order": {"_key": "asc"},
                    "size": 7000,
                },
                "aggs": {
                    "hits": {
                        "top_hits": {
                            "_source": [
                                "site_type",
                                "site_code",
                                "facility_code",
                                "location",
                            ],
                            "size": 1,
                        },
                    },
                },
            },
        },
        "size": 0,
        "query": {
            "query_string": {
                "query": query,
            },
        },
    }

    # Uses requests to grab metadata from arm.gov.
    response = requests.get(
        'https://adc.arm.gov/elastic/metadata/_search', headers=headers, json=json_data
    )
    # Loads the text to a dictionary
    response_dict = json.loads(response.text)

    # Searches dictionary for the site, facility and coordinate information.
    coord_dict = {}
    # Loop through each facility.
    for i in range(len(response_dict['aggregations']['distinct_facility_code']['buckets'])):
        site_info = response_dict['aggregations']['distinct_facility_code']['buckets'][i]['hits'][
            'hits'
        ]['hits'][0]['_source']
        site = site_info['site_code']
        facility = site_info['facility_code']
        # Some sites do not contain coordinate information, return None if that is the case.
        if site_info['location'] is None:
            coords = {'latitude': None, 'longitude': None}
        else:
            lat, lon = site_info['location'].split(',')
            lat = float(lat)
            lon = float(lon)
            coords = {'latitude': lat, 'longitude': lon}
        coord_dict.setdefault(site + ' ' + facility, coords)

    return coord_dict
