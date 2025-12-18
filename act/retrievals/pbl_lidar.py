"""
Functions for planetary boundary layer height estimation
related calculations from lidar

"""
import numpy as np
import xarray as xr
from scipy.signal import find_peaks


def calculate_gradient_pbl(ds, parm="beta_att", dis_parm="range", min_height=100, smooth_dis=5):
    """
    Estimation of the Planetary Boundary Layer (PBL) height from a backscatter LIDAR
    through a gradient method, where the PBL height is identified through the
    sharpest negative gradient.

    Note:
    This retrieval method should be applied under a cloud-free, well-mixed PBL condition.

    It is not expected perform well in cloud capped boundary layers.
    Additional PRs will be included within the near future to address more PBL
    environmental conditions.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the zenith-pointing remote sensing data.
    parm : str
        Variable in the dataset to compute gradient on (e.g., attenuated backscatter).
    dis_parm : str
        Distance-from-instrument coordinate (e.g., 'range' or 'height').
    min_height : float
        Minimum allowed PBL height in meters.
    smooth_dis : int
        Number of bins to average vertical profile over to smooth data

    Returns
    -------
    ds : xarray.Dataset
        Dataset with a new variable `pbl_gradient` containing PBL heights.

    References
    ----------
    Hayden, K. L. et al. (1997): The vertical chemical and meteorological
        structure of the boundary layer in the Lower Fraser Valley during
        Pacific ’93. Atmospheric Environment, 31, 2089–2105,
        https://doi.org/10.1016/S1352-2310(96)00300-7.

    Li, H., Yang, Y., Hu, X.M., Huang, Z., Wang, G., Zhang, B., Zhang, T. (2017).
        Evaluation of retrieval methods of daytime convective boundary layer
        height based on lidar data. J. Geophys. Res. 122, 4578–4593.
        https://doi.org/10.1002/2016JD025620

    Wang, Y.-C., Wang, S.-H., Lewis, J. R., Chang, S.-C., & Griffith, S. M.
        (2021). Determining Planetary Boundary Layer Height by Micro-pulse Lidar
        with Validation by UAV Measurements. Aerosol and Air Quality Research,
        21 (5), 200336. Retrieved 2025-10-15, from
        https://aaqr.org/articles/aaqr-20-06-oa-0336
    """
    # smooth the data within the range bins (~20m bins)
    smoothed = ds[parm].rolling({dis_parm: smooth_dis}, center=True).mean()

    # Loop over time to find the sharpest negative gradient
    pbl_heights = []

    for t in range(len(ds["time"].values)):
        profile = smoothed.isel(time=t).values  # 1D backscatter profile
        height = smoothed[dis_parm].values  # 1D height coordinate

        # Compute first derivative
        p_grad = np.gradient(profile, height)

        # Find the first negative gradient
        indice = next(i for i, x in enumerate(p_grad) if x < 0)

        # Choose the first peak above a certain altitude (e.g., ignore surface noise)
        if height[indice] > min_height:
            pbl_heights.append(height[indice])
        else:
            pbl_heights.append(np.nan)

    # Add result to dataset
    ds = ds.assign(pbl_gradient=xr.DataArray(pbl_heights, dims="time"))
    ds['pbl_gradient'].attrs[
        "description"
    ] = "Planetary Boundary Layer Estimate via Gradient Method"
    ds['pbl_gradient'].attrs["input_parameter"] = parm
    if hasattr(ds[dis_parm], "units"):
        ds['pbl_gradient'].attrs["units"] = ds[dis_parm].attrs["units"]
    else:
        ds['pbl_gradient'].attrs["units"] = "meters"

    return ds


def calculate_modified_gradient_pbl(
    ds, parm="beta_att", dis_parm="range", min_height=100, threshold=1e-3, smooth_dis=5
):
    """
    Estimation of the Planetary Boundary Layer (PBL) height from a backscatter LIDAR
    through a modified gradient method, where the first significant inflection point
    within the profile is identified rather than the traditional sharpest negative gradient.

    Also conforms to the depolarization ratio threshold PBL height estimate when
    the `parm` input is properly selected.

    Note:
    This retrieval method should be applied under a cloud-free, well-mixed PBL condition.

    It is not expected perform well in cloud capped boundary layers.
    Additional PRs will be included within the near future to address more PBL
    environmental conditions.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the zenith-pointing remote sensing data.
    parm : str
        Variable in the dataset to compute gradient on (e.g., attenuated backscatter).
    dis_parm : str
        Distance-from-instrument coordinate (e.g., 'range' or 'height').
    min_height : float
        Minimum allowed PBL height in meters.
    threshold : float
        Prominence value to use within scipy.signal.find_peaks
    smooth_dis : int
        Number of bins to average vertical profile over to smooth data

    Returns
    -------
    ds : xarray.Dataset
        Dataset with a new variable `pbl_mod_gradient` containing PBL heights.

    References
    ----------
    Satheesh, A. R., Warner, G., Cai, J., Juliano, T., O'Brien, J. R.,
        & Wagner, T. (2025). Boundary Layer in Multiple Places (BLIMP)
        (v2025.05.29). Zenodo. https://doi.org/10.5281/zenodo.15545989

    Jackson, R., O’Brien, J., Wang, J., Fytanidis, D., Muradyan, P.,
        Grover, M., Raut, B., Collis, S., Tuftedal, M., Anderson, G.,
        agner, T. J., Nesbitt, S., Tan. H., Wefer, D., & Hammond, M. (2025).
        The thermodynamic and kinematic structure of the planetary boundary
        layer for a summer lake breeze day in Chicago. Journal of Geophysical
        Research: Atmospheres, in preparation.
    """
    # smooth the data within the range bins (~20m bins)
    smoothed = ds[parm].rolling({dis_parm: smooth_dis}, center=True).mean()

    # Loop over time to get peaks in second derivative
    pbl_heights = []

    for t in range(len(ds["time"].values)):
        profile = smoothed.isel(time=t).values  # 1D backscatter profile
        height = smoothed[dis_parm].values  # 1D height coordinate

        # Compute first and second derivatives
        d1 = np.gradient(profile, height)
        d2 = np.gradient(d1, height)

        # Invert second derivative to find local minima
        # These can indicate PBL top or inversion-like transitions
        peaks, _ = find_peaks(-d2, distance=10, prominence=threshold)

        if len(peaks) > 0:
            # Choose the first peak above a certain altitude (e.g., ignore surface noise)
            valid_peaks = [p for p in peaks if height[p] > min_height]
            if valid_peaks:
                pbl_heights.append(height[valid_peaks[0]])
            else:
                pbl_heights.append(np.nan)
        else:
            pbl_heights.append(np.nan)

    # Add result to dataset
    ds = ds.assign(pbl_mod_gradient=xr.DataArray(pbl_heights, dims="time"))
    ds['pbl_mod_gradient'].attrs[
        "description"
    ] = "Planetary Boundary Layer Estimate via modified gradient method"
    ds['pbl_mod_gradient'].attrs["input_parameter"] = parm
    ds['pbl_mod_gradient'].attrs["prominence_threshold"] = threshold
    if hasattr(ds[dis_parm], "units"):
        ds['pbl_mod_gradient'].attrs["units"] = ds[dis_parm].attrs["units"]
    else:
        ds['pbl_mod_gradient'].attrs["units"] = "meters"

    return ds
