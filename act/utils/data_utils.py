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


def get_missing_value(self, variable, default=-9999, add_if_missing_in_obj=False,
    use_FillValue=False, nodefault=False):
    '''Method to get missing value from missing_value or _FillValue attribute.
    Works well with catching errors and allows for a default value when a missing
    value is not listed in the object.

    Parameters
    ----------
    variable : str
        Variable name to use for getting missing value. 
    default : int or float
        Default value to use if missing value attribute is not in data object
    add_if_missing_in_obj : bool
        Boolean to add to object if does not exist. Default is False.
    use_FillValue : bool
        Boolean to use _FillValue instead of missing_value. If missing_value
        does exist and _FillValue does not with add_if_missing_in_obj set to 
        True, will add _FillValue set to missing_value value. Default is False.
    nodefault : bool
        Option to use this to check if the varible has a missing value set and do not 
        want to get default as retun. If the missing value is found will return, 
        else will return None.

    Returns
    -------
    missing : scalar int or float (or None)
        Value used to indicate missing value matching type of data or None if 
        nodefault keyword set to True.

    Examples
    --------
    >>> missing = dq_object.clean.get_missing_value('temp_mean')
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
            missing = self._data_object[variable].attrs[att]
            in_object = True
            break
        except (AttributeError, KeyError):
            missing = default

    # Check if do not want a default value retured and a value
    # was not fund.
    if nodefault == True and in_object == False:
        missing = None
        return missing

    # Check data type and try to match missing_value to the data type of data
    try:
        missing = self._data_object[variable].data.dtype.type(missing)
    except KeyError:
        pass
    except AttributeError:
        print(('--- AttributeError: Issue trying to get data type '+
            'from "{}" data ---').format(variable))

    # If requested add missing value to object
    if add_if_missing_in_obj and not in_object:
        try:
            self._data_object[variable].attrs[missing_atts[0]] = missing
        except KeyError:
            print(('---  KeyError: Issue trying to add "{}" '+
                'attribute to "{}" ---').format(missing_atts[0],variable))

    return missing
