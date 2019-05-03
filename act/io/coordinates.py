"""
==================
act.io.coordinates
==================

This procedure has differing routines for manipulating the coordinate systems
of input datasets for easier data visualization.

"""
import xarray as xr


def assign_coordinates(ds, coord_list):
    """
    This procedure will create a new ACT dataset whose coordinates are designated
    to be the variables in a given list. This helps make data slicing via xarray
    and visualization easier.

    Parameters
    ----------
    ds: ACT Dataset
        The ACT Dataset to modify the coordinates of.
    coord_list: list
        The list of variables to assign as coordinates

    Returns
    -------
    new_ds: ACT Dataset
        The new ACT Dataset with the coordinates assigned to be the given variables.
    """

    # Check to make sure that user assigned valid entries for coordinates
    
    for coord in coord_list:
        if not coord in ds.variables.keys():
            raise KeyError(coord + " is not a variable in the Dataset.")




