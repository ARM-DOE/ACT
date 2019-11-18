import os

import numpy as np
import pandas as pd


def calculate_aospurge_times(obj, threshold=None,
                             txt_path=None):

    """
    Method to calculate times where the AOS purge system is active. It will
    print out the start and end times and has the option to save those time
    ranges to a .txt file. When there is a high amount of contamination in the
    air, the AOS activates its purge system to flush the contaminated air from
    the system. During this time, the AOS is not sampling ambient air and a DQR
    should be written to include the time ranges when the purge system is
    active.

    Parameters
    ----------
    obj : ACT Object
        Xarray Dataset as read by ACT where data are stored.
    threshold : int
        Minimum number of data points between separate start and end time
        ranges. Will default to 1 minute.
    txt_path : str
        Path in which to save .txt files of start and end times. If set to none
        will not save a text file.

    Returns
    -------
    time_strings : list
        List of tuples with the first element as the start time and the second
        element as the end time.

    Examples
    --------
    This example will return the times used for unit testing.

    .. code-block:: python

        import act

        obj = act.io.armfiles.read_netcdf(
            act.tests.sample_files.EXAMPLE_AOS_PURGE)

        times = act.retrievals.purge_times.calculate_aospurge_times(obj)

    """
    # Get date from object
    date = obj.attrs['file_dates'][0]

    # Set default threshold value of 1 minute if None given.
    if not threshold:
        threshold = len(obj['time'].values) / 1440

    # There is a variable in the aospurge data called stack_purge_state
    # where a value of 1 indicates that the AOS is not sampling ambient
    # air. Find the times where this is valid.
    try:
        stack = obj['stack_purge_state'].values
    except ValueError:
        return
    # Getting time indices where purge is active
    time_indices = np.where(stack == 1)[0]

    # Calculate if there are multiple instances per day of purge
    # Find instances in the array of time indices where difference > 1
    time_diff = np.diff(time_indices)

    # Split the array of time indices where blower is on for at least
    # 60 seconds
    splits = np.where(time_diff > threshold)
    splits = splits[0] + 1

    # If no indices where purge occurs then exit
    if len(time_indices) == 0:
        print('No purge occurs on ' + date)
        return
    else:
        time_indices = np.split(time_indices, splits)

    dt_times = []
    for time in time_indices:
        dt_times.append((obj['time'].values[time[0]],
                        obj['time'].values[time[-1]]))

    # Convert the datetimes to strings
    time_strings = []
    for st, et in dt_times:
        start_time = pd.to_datetime(str(st)).strftime('%Y-%m-%d %H:%M:%S')
        end_time = pd.to_datetime(str(et)).strftime('%Y-%m-%d %H:%M:%S')
        time_strings.append((start_time, end_time))
        # Print times to screen
        print('AOSPURGE Begins at: ' + start_time, flush=True)
        print('AOSPURGE Ends at: ' + end_time, flush=True)

    if txt_path:
        # Save to .txt file
        if os.path.exists(txt_path) is False:
            os.mkdir(txt_path)
        with open(txt_path + '/dqrtimes_' + date + '.txt', 'w') as \
                text_file:
            for st, et in time_strings:
                text_file.write('%s, ' % st)
                text_file.write('%s \n' % et)
    return time_strings
