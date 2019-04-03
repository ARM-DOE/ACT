import numpy as np
import scipy.stats as stats
import xarray as xr


def add_in_nan(time, data):
    """
    This procedure adds in NaNs for given time periods in time when there is no
    corresponding data available. This is useful for timeseries that have
    irregular gaps in data.

    Parameters:
    -----------
    time: 1D array of np.datetime64
        List of times in the timeseries
    data: 1 or 2D array
        Array containing the data. The 0 axis corresponds to time.

    Returns:
    --------
    d_time: xarray DataArray
        The xarray DataArray containing the new times at regular intervals.
        The intervals are determined by the mode of the timestep in *time*.
    d_data: xarray DataArray
        The xarray DataArray containing the NaN-filled data.
    """

    diff = time.diff(dim='time', n=1) / np.timedelta64(1, 's')
    mode = stats.mode(diff).mode[0]
    index = np.where(diff.values > 2. * mode)
    d_data = np.asarray(data)
    d_time = np.asarray(time)

    offset = 0
    for i in index[0]:
        n_obs = np.floor(
            (time[i + 1] - time[i]) / mode / np.timedelta64(1, 's'))
        time_arr = [
            d_time[i + offset] + np.timedelta64(int((n + 1) * mode), 's')
            for n in range(int(n_obs))]
        S = d_data.shape
        if len(S) == 2:
            data_arr = np.empty([len(time_arr), S[1]])
        else:
            data_arr = np.empty([len(time_arr)])
        data_arr[:] = np.nan

        d_time = np.insert(d_time, i + 1 + offset, time_arr)
        d_data = np.insert(d_data, i + 1 + offset, data_arr, axis=0)
        offset += len(time_arr)

    d_time = xr.DataArray(d_time)
    d_data = xr.DataArray(d_data)

    return d_time, d_data


def get_missing_value(data_object, variable, default=-9999, add_if_missing_in_obj=False,
        use_FillValue=False):
    '''Function to get missing value from missing_value or _FillValue attribute.
    Works well with catching errors and allows for a default value when a missing
    value is not listed in the object.

    Parameters
    ----------
    data_object : obj
        Data object to search.
    variable : str
        Variable name to use for getting missing value. 
    default : int or float or None
        Default value to use if missing value attribute is not in data_object.
        If no value should be retuned if missing_value or _FillValue is not 
        set as attribute in object, set default=None.
    add_if_missing_in_obj : bool
        Boolean to add to object if does not exist.
    use_FillValue : bool
        Boolean to use _FillValue instead of missing_value. If missing_value
        does exist and _FillValue does not with add_if_missing_in_obj set to 
        True, will add _FillValue set to missing_value value.

    Returns
    -------
    missing : scalar int or float
        Value used to indicate missing value matching type of data.

    Examples
    --------
    >>> missing = get_missing_value(dq_object, 'temp_mean')
    >>> missing
    -9999.0

    '''
    in_object = False
    if use_FillValue:
        missing_atts = ['_FillValue','missing_value']
    else:
        missing_atts = ['missing_value','_FillValue']

    for att in missing_atts:
        try:
            missing = data_object[variable].attrs[att]
            in_object = True
            break
        except (AttributeError, KeyError):
            missing = default

    # Check data type and try to match missing_value to the data type of data
    try:
        missing = data_object[variable].data.dtype.type(missing)
    except Exception as error:
        pass

    # If requested add missing value to object
    if add_if_missing_in_obj and not in_object:
        try:
            data_object[variable].attrs[missing_atts[0]] = missing
        except Exception as error:
            pass

    return missing
