import numpy as np
import scipy.stats as stats
import xarray as xr


def add_in_nan(time, data):
    """
    Procedure add_in_nan
    --------------------
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
