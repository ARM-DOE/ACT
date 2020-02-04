"""
===============
act.io.csvfiles
===============
This module contains I/O operations for loading csv files.

"""

# import standard modules
import pandas as pd

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
    # Read data using pandas read_csv
    arm_ds = pd.read_csv(filename, sep=sep, names=column_names,
                         skipfooter=skipfooter, engine=engine, **kwargs)

    # Set Coordinates if there's a variable date_time
    if 'date_time' in arm_ds:
        arm_ds.date_time = arm_ds.date_time.astype('datetime64')
        arm_ds.time = arm_ds.date_time
        arm_ds = arm_ds.set_index('time')

    # Convert to xarray DataSet
    arm_ds = arm_ds.to_xarray()

    # Set additional variables
    # Since we cannot assume a standard naming convention setting
    # file_date and file_time to the first time in the file
    x_coord = arm_ds.coords.to_index().values[0]
    if isinstance(x_coord, str):
        x_coord_dt = pd.to_datetime(x_coord)
        arm_ds.attrs['_file_dates'] = x_coord_dt.strftime('%Y%m%d')
        arm_ds.attrs['_file_times'] = x_coord_dt.strftime('%H%M%S')

    # Check for standard ARM datastream name, if none, assume the file is ARM
    # standard format.
    is_arm_file_flag = check_arm_standards(arm_ds)
    if is_arm_file_flag.NO_DATASTREAM is True:
        arm_ds.attrs['_datastream'] = '.'.join(filename.split('/')[-1].split('.')[0:2])

    # Add additional attributes, site, standards flag, etc...
    arm_ds.attrs['_site'] = str(arm_ds.attrs['_datastream'])[0:3]

    arm_ds.attrs['_arm_standards_flag'] = is_arm_file_flag

    return arm_ds
