"""Shared inputs for transform behavior and interface tests."""

import numpy as np
import xarray as xr

MISSING = -9999.0


def _da(values, coord_name="time", coord=None, name="temp"):
    if coord is None:
        coord = np.arange(len(values), dtype=float)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        coords={coord_name: coord},
        dims=[coord_name],
        name=name,
    )
