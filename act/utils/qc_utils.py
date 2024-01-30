"""
Functions containing utilities for quality control which
may or may not be program dependent

"""

import os

import numpy as np
import pandas as pd


def calculate_dqr_times(
    ds, variable=None, txt_path=None, threshold=None, qc_bit=None, return_missing=True
):
    """
    Function to retrieve start and end times of missing or bad data. Function
    will retrieve start and end time strings in a format that the DQR
    submission tool can read, print them to the console, and save to a .txt
    file if desired.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset as read by ACT where data are stored.
    variable : str or list of str
        Data variable(s) in the dataset to check. Can check multiple variables.
    txt_path : str
        Full path to directory in which to save .txt files with start and end
        times. If directory doesn't exist the program will create it. If set
        to None then no .txt files will be created. Default is None.
    threshold : int
        Threshold of number of data points to trigger start and end time
        calculations. Value is interpreted differently depending on the
        resolution of the data in the specified files. For example, if data is
        1-minute data, a threshold of 30 would mean flagged data more than 30
        minutes apart would show up as two separate time ranges; if data is 30-
        minute data, a threshold of 2 would mean flagged data more than 60 mins
        apart would show up as two separate time ranges.
    qc_bit : int
        Bit number to choose if finding times for bad data. If set then will
        override searching for missing data. Default is None.
    return_missing : bool
        Specifies whether or not to return times of data flagged as missing. If
        set to False, will return times of bad data. Default is True.

    Returns
    -------
    time_strings : list
        List of tuples with the first element as the start time and the second
        element as the end time.

    """
    # Determine if searching for either bad or missing data
    if not return_missing:
        return_bad = True
        return_missing = False
    else:
        return_bad = False

    # If qc bit set then make sure searching for bad data
    if qc_bit:
        return_bad = True
        return_missing = False

    # Clean files. Converts from ARM to CF standards
    ds.clean.cleanup(cleanup_arm_qc=True, handle_missing_value=True, link_qc_variables=True)
    date = ds.attrs['_file_dates'][0]
    datastream = ds.attrs['_datastream']

    # Make variable instance a list always
    if variable is not None:
        if not isinstance(variable, list):
            variable = [variable]
    else:
        variable = []

    # Make sure that threshold number is an integer. If not convert to
    # closest integer
    if threshold is not None:
        if not isinstance(threshold, int):
            int(round(threshold))
    else:
        print('You must specify a threshold for separating ranges of' + ' flagged data')
        return

    # If return_missing then search for indices where data is equal to
    # missing value.
    if return_missing:
        for var in variable:
            # Get indices where data is being flagged as missing
            idx = np.where(np.isnan(ds[var].values))[0]

            # Find where there are gaps in flagged data
            time_diff = np.diff(idx)

            # Find indices in flagged data where gaps occur
            splits = np.where(time_diff > threshold)
            splits = splits[0] + 1

            # If no bad indices then exit
            if len(idx) == 0:
                print(f'No missing data for {var} on ' + date)
                continue
            else:
                idx = np.split(idx, splits)
            # Now that we have all of the stretches of bad flags, get
            # corresponding datetimes from time data
            dt_times = []

            for time in idx:
                # If there is only one flagged data point skip
                if len(time) < 2:
                    pass
                else:
                    dt_times.append((ds['time'].values[time[0]], ds['time'].values[time[-1]]))
            # Convert the datetimes to strings
            time_strings = []
            for st, et in dt_times:
                start_time = pd.to_datetime(str(st)).strftime('%Y-%m-%d %H:%M:%S')
                end_time = pd.to_datetime(str(et)).strftime('%Y-%m-%d %H:%M:%S')
                time_strings.append((start_time, end_time))
                # Print times to screen
                print(f'Missing Data for {var} begins at: ' + start_time)
                print(f'Missing Data for {var} ends at: ' + end_time)
            if txt_path:
                _write_dqr_times_to_txt(datastream, date, txt_path, var, time_strings)
            return time_strings

    # If return_bad then search for times in the corresponding qc variable
    # where the flags are tripped
    if return_bad:
        for var in variable:
            # If a1 level data, return.
            if 'a1' in ds.attrs['data_level']:
                print('No QC is present in a1 level.')
                return
            # Get QC data from corresponding QC variable
            qc_var = 'qc_' + var
            try:
                qc_data = ds[qc_var].values
            except KeyError:
                print('Unable to calculate start and end times for bad data')
                continue
            # Make sure qc bit is an integer
            if not isinstance(qc_bit, int):
                raise TypeError('QC bit must be an integer')
            # Get indices where data is being flagged for given qc bit
            idx = np.where(qc_data == 2 ** (qc_bit - 1))[0]

            # Find where there are gaps in flagged data
            time_diff = np.diff(idx)

            # Find indices in flagged data where gaps occur
            splits = np.where(time_diff > threshold)
            splits = splits[0] + 1

            # If no bad indices then exit
            if len(idx) == 0:
                print('No bad data on ' + date + ' for selected QC bit for' + ' variable ' + var)
                continue
            else:
                idx = np.split(idx, splits)
            # Now that we have all of the stretches of bad flags, get
            # corresponding datetimes from time data
            dt_times = []

            for time in idx:
                # If there is only one flagged data point skip
                if len(time) < 2:
                    pass
                else:
                    dt_times.append((ds['time'].values[time[0]], ds['time'].values[time[-1]]))
            # Convert the datetimes to strings
            time_strings = []
            for st, et in dt_times:
                start_time = pd.to_datetime(str(st)).strftime('%Y-%m-%d %H:%M:%S')
                end_time = pd.to_datetime(str(et)).strftime('%Y-%m-%d %H:%M:%S')
                time_strings.append((start_time, end_time))
                # Print times to screen
                print(f'Bad Data for {var} Begins at: ' + start_time)
                print(f'Bad Data for {var} Ends at: ' + end_time)
            if txt_path:
                _write_dqr_times_to_txt(datastream, date, txt_path, var, time_strings)
            return time_strings


def _write_dqr_times_to_txt(datastream, date, txt_path, variable, time_strings):
    """
    Writes flagged data time range(s) to a .txt file. The naming convention is
    dqrtimes_datastream.date.txt.

    Parameters
    ----------
    datastream : str
        ARM datastream name (ie, sgpmetE13.b1).
    date : str
        Date of time range(s) being written.
    txt_path : str
        Base path of where the .txt files are being written.
    variable : str
        Name of variable being flagged.
    time_strings : list of tuples
        List of every start and end time to be written.

    """
    print(
        'Writing data to text file for '
        + datastream
        + ' '
        + variable
        + ' on '
        + date
        + ' at '
        + txt_path
        + '...',
        flush=True,
    )
    full_path = txt_path + '/' + datastream
    if os.path.exists(full_path) is False:
        os.mkdir(full_path)
    with open(
        full_path + '/dqrtimes_' + datastream + '.' + date + '.' + variable + '.txt',
        'w',
    ) as text_file:
        for st, et in time_strings:
            text_file.write('%s, ' % st)
            text_file.write('%s \n' % et)
