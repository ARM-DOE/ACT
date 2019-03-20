# import standard modules
import glob
import xarray as xr
import warnings

from .dataset import ACTAccessor

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
    act_obj : Object
        ACT dataset

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

    arm_ds.act.file_dates = file_dates
    arm_ds.act.file_times = file_times
    if not 'datastream' in arm_ds.attrs.keys():
        warnings.warn(UserWarning, "ARM standards require that the datastream name be defined, currently using a default" +
                           " of act_datastream.")
        arm_ds.act.datastream = "act_datastream"
    else:
        arm_ds.act.datastream = arm_ds.attrs["datastream"]
    arm_ds.act.site = str(arm_ds.act.datastream)[0:3]

    return arm_ds
