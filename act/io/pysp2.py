try:
    import pysp2

    PYSP2_AVAILABLE = True
except ImportError:
    PYSP2_AVAILABLE = False


def read_hk_file(file_name):
    """
    This procedure will read in an SP2 housekeeping file and then
    store the timeseries data into a pandas DataFrame.

    Parameters
    ----------
    file_name: str
        The file name to read in

    Returns
    -------
    hk_ds: xarray.Dataset
        The housekeeping information in an xarray Dataset
    """
    if PYSP2_AVAILABLE:
        return pysp2.io.read_hk_file(file_name)
    else:
        raise ModuleNotFoundError(
            'PySP2 must be installed in order to read SP2 data and housekeeping files.'
        )


def read_sp2(file_name, debug=False, arm_convention=True):
    """
    Loads a binary SP2 raw data file and returns all of the wave forms
    into an xarray Dataset.

    Parameters
    ----------
    file_name: str
        The name of the .sp2b file to read.
    debug: bool
        Set to true for verbose output.
    arm_convention: bool
        If True, then the file name will follow ARM standard naming conventions.
        If False, then the file name follows the SP2 default naming convention.

    Returns
    -------
    ds: xarray.Dataset
        The xarray Dataset containing the raw SP2 waveforms for each particle.
    """
    if PYSP2_AVAILABLE:
        return pysp2.io.read_sp2(file_name, debug, arm_convention)
    else:
        raise ModuleNotFoundError(
            'PySP2 must be installed in order to read SP2 data and housekeeping files.'
        )


def read_sp2_dat(file_name, type):
    """
    This reads the .dat files that generate the intermediate parameters used
    by the Igor processing. Wildcards are supported.

    Parameters
    ----------
    file_name: str
        The name of the file to save to. Use a wildcard to open multiple files at once.
    type: str
        This parameter must be one of:
            'particle': Load individual particle timeseries from .dat file
            'conc': Load timeseries of concentrations.
    Returns
    -------
    ds: xarray.Dataset
        The xarray dataset to store the parameters in.
    """
    if PYSP2_AVAILABLE:
        return pysp2.io.read_dat(file_name, type)
    else:
        raise ModuleNotFoundError(
            'PySP2 must be installed in order to read SP2 data and housekeeping files.'
        )
