"""
This module contains I/O operations for loading csv files.

"""

import pandas as pd
import pathlib

from .armfiles import check_arm_standards


def read_csv(filename, sep=',', engine='python', column_names=None,
             skipfooter=0, **kwargs):

    """
    Returns an `xarray.Dataset` with stored data and metadata from user-defined
    query of CSV files.

    Parameters
    ----------
    filenames : str or list
        Name of file(s) to read.
    sep : str
        The separator between columns in the csv file.
    column_names : list or None
        The list of column names in the csv file.
    verbose : bool
        If true, will print if a file is not found.

    Additional keyword arguments will be passed into pandas.read_csv.

    Returns
    -------
    act_obj : Object
        ACT dataset. Will be None if the file is not found.

    Examples
    --------
    This example will load the example sounding data used for unit testing:

    .. code-block:: python

        import act
        the_ds, the_flag = act.io.csvfiles.read(
            act.tests.sample_files.EXAMPLE_CSV_WILDCARD)

    """

    # Convert to string if filename is a pathlib or not a list
    if isinstance(filename, (pathlib.PurePath, str)):
        filename = [str(filename)]

    if isinstance(filename, list) and isinstance(filename[0], pathlib.PurePath):
        print('filename')
        filename = [str(ii) for ii in filename]

    # Read data using pandas read_csv one file at a time and append to
    # list. Then concatinate the list into one pandas dataframe.
    li = []
    for fl in filename:
        df = pd.read_csv(fl, sep=sep, names=column_names,
                         skipfooter=skipfooter, engine=engine, **kwargs)
        li.append(df)

    if len(li) == 1:
        df = li[0]
    else:
        df = pd.concat(li, axis=0, ignore_index=True)

    # Set Coordinates if there's a variable date_time
    if 'date_time' in df:
        df.date_time = df.date_time.astype('datetime64')
        df.time = df.date_time
        df = df.set_index('time')

    # Convert to xarray DataSet
    ds = df.to_xarray()

    # Set additional variables
    # Since we cannot assume a standard naming convention setting
    # file_date and file_time to the first time in the file
    x_coord = ds.coords.to_index().values[0]
    if isinstance(x_coord, str):
        x_coord_dt = pd.to_datetime(x_coord)
        ds.attrs['_file_dates'] = x_coord_dt.strftime('%Y%m%d')
        ds.attrs['_file_times'] = x_coord_dt.strftime('%H%M%S')

    # Check for standard ARM datastream name, if none, assume the file is ARM
    # standard format.
    is_arm_file_flag = check_arm_standards(ds)
    if is_arm_file_flag == 0:

        ds.attrs['_datastream'] = '.'.join(filename[0].split('/')[-1].split('.')[0:2])

    # Add additional attributes, site, standards flag, etc...
    ds.attrs['_site'] = str(ds.attrs['_datastream'])[0:3]
    ds.attrs['_arm_standards_flag'] = is_arm_file_flag

    return ds
