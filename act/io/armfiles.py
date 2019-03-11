# import standard modules
import glob
import xarray as xr


def read_netcdf(filenames, variables=None):

    """
    Returns `xarray.Dataset` with stored data and metadata from a user-defined
    query of ARM-standard netCDF files from a single datastream.

    Parameters
    ----------
    filenames : str or list
        Name of file(s) to read

    Other Parameters
    ----------
    variables : list, optional
        List of variable name(s) to read

    Returns
    ----------
    arm_obj : Object
        Xarray data object

    """

    file_dates = []
    file_times = []
    arm_ds = xr.open_mfdataset(filenames, parallel=True, concat_dim='time')

    # Adding support for wildcards
    if isinstance(filenames, str):
        filenames = glob.glob(filenames)

    filenames.sort()
    for n, f in enumerate(filenames):
        file_dates.append(f.split('.')[-3])
        file_times.append(f.split('.')[-2])

    arm_ds['file_dates'] = file_dates
    arm_ds['file_times'] = file_times
    arm_ds['ds'] = (filenames[0].split('.')[0]).split('/')[-1]
    arm_ds['site'] = str(arm_ds['ds'].values)[0:3]

    return arm_ds
