"""
Functions for planetary boundary layer height estimation
related calculations from lidar

"""

import numpy as np
import xarray as xr
from scipy.signal import find_peaks

try:
    import pywt

    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False


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


def calculate_wavelet_pbl(
    ds,
    var_name='wind_speed',
    range_name='height',
    scale=60.0,
    continuity_window=2,
    min_height=100,
    max_height=None,
):
    """
    Estimation of the Planetary Boundary Layer (PBL) height from a ceilometer
    or Doppler lidar through a Haar wavelet covariance transform. The dataset
    is averaged into 5-minute periods, and each vertical profile is decomposed
    with a Haar wavelet. The PBL height at each time is taken to be the range
    at which the wavelet approximation coefficients show their sharpest
    transition. A continuity check then replaces PBL height estimates that
    jump more than 150 m above their neighbors with the local baseline.

    Note:
    This retrieval method should be applied under a cloud-free, well-mixed PBL condition.

    It is not expected perform well in cloud capped boundary layers.
    Additional PRs will be included within the near future to address more PBL
    environmental conditions.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the zenith-pointing ceilometer or Doppler lidar data.
    var_name : str
        Variable in the dataset to compute the wavelet transform on (e.g.,
        backscatter intensity or vertical velocity).
    range_name : str
        Name of the range/height coordinate in the dataset.
    scale : float
        Approximate spatial scale, in the same units as range_name, over which
        the Haar wavelet decomposition is performed. This sets the decomposition level.
    continuity_window : int
        Number of neighboring time steps on each side of a given time to
        average over when checking for, and smoothing out, discontinuous PBL
        height estimates.
    min_height : float
        Minimum allowed PBL height in the units of range_name. Excludes
        near-surface noise from the search for the sharpest transition.
    max_height : float or None
        Maximum allowed PBL height in the units of range_name. Use this to
        exclude elevated cloud or aerosol layers above the PBL from the
        search. If None, no upper bound is applied.

    Returns
    -------
    ds : xarray.Dataset
        Dataset resampled to 5-minute periods with new variables
        `wavelet_backscatter`, containing the Haar wavelet approximation
        coefficients, and `pbl_wavelet`, containing the estimated PBL heights.

    References
    ----------
    Brooks, I. M. (2003). Finding boundary layer top using
        wavelet covariance transform. Journal of Atmospheric and Oceanic
        Technology, 20(8), 1092-1105.
        https://doi.org/10.1175/1520-0426(2003)20%3C1092:FBLTUB%3E2.0.CO;2

    Cohn, S. A., & Angevine, W. M. (2000). Boundary layer height and
        entrainment zone thickness measured by lidars and wind-profiling
        radars. Journal of Applied Meteorology, 39(8), 1233-1247.
        https://doi.org/10.1175/1520-0450(2000)039%3C1233:BLHAEZ%3E2.0.CO;2
    """
    if not PYWAVELETS_AVAILABLE:
        raise ImportError('PyWavelets needs to be installed to use this feature.')

    ds = ds.resample(time='5min').mean()
    range_resolution = ds[range_name].values[1] - ds[range_name].values[0]
    level = int(scale / range_resolution) - 1

    coeffs = pywt.wavedec(ds[var_name].values, 'haar', level=level)
    cA = coeffs[0]

    resampled_range = ds[range_name].values[:: 2**level]
    resampled_time = ds.time.values

    ds['resampled_range'] = resampled_range
    ds['resampled_time'] = resampled_time
    ds = ds.set_coords(['resampled_range', 'resampled_time'])
    ds['wavelet_backscatter'] = (('resampled_time', 'resampled_range'), cA)

    range_mask = ds.resampled_range >= min_height
    if max_height is not None:
        range_mask = range_mask & (ds.resampled_range <= max_height)
    wavelet_valid = ds.wavelet_backscatter.where(range_mask, drop=True)

    max_gradient = wavelet_valid.diff('resampled_range').max('resampled_range')
    pbl_heights = []
    for t in range(len(ds.resampled_time)):
        profile = wavelet_valid.isel(resampled_time=t)
        try:
            pbl_height = profile.where(
                profile.diff('resampled_range') == max_gradient.isel(resampled_time=t),
                drop=True,
            ).resampled_range.values[0]
        except IndexError:
            pbl_height = np.nan
        pbl_heights.append(pbl_height)

    pbl_heights = np.array(pbl_heights, dtype=float)
    for i in range(continuity_window, len(pbl_heights) - continuity_window):
        neighbors = np.concatenate(
            [
                pbl_heights[i - continuity_window : i],
                pbl_heights[i + 1 : i + continuity_window + 1],
            ]
        )
        baseline = np.nanmean(neighbors)
        if pbl_heights[i] > baseline + 150:
            pbl_heights[i] = baseline

    ds['pbl_wavelet'] = xr.DataArray(pbl_heights, dims='resampled_time')
    ds['pbl_wavelet'].attrs[
        'description'
    ] = 'Planetary Boundary Layer Estimate via Haar wavelet covariance transform'
    ds['pbl_wavelet'].attrs['input_parameter'] = var_name
    if hasattr(ds[range_name], 'units'):
        ds['pbl_wavelet'].attrs['units'] = ds[range_name].attrs['units']
    else:
        ds['pbl_wavelet'].attrs['units'] = 'meters'

    return ds
