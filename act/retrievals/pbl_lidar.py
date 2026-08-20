"""
Functions for planetary boundary layer height estimation
related calculations from lidar

"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks

try:
    import pywt

    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import acf
except:
    acf = None


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


def calculate_tucker_method_pbl(
    ds,
    velocity="radial_velocity",
    dis_parm="range",
    interval="30min",
    threshold=0.08,
    noise_variance_threshold=0.2,
    min_gate_height=100,
):
    """
    Estimation of the Planetary Boundary Layer (PBL) height from Doppler lidar
    radial velocity using the turbulence component of the Tucker et al. (2009)
    method.

    For each averaging interval and range gate, the lag-1 autocorrelation of
    the radial velocity is used to separate the raw velocity variance into an
    atmospheric (turbulent) component and an instrument noise component.
    Since instrument noise is uncorrelated in time, it inflates the lag-0
    variance but not the lag-1 autocorrelation, so the noise variance is
    estimated as (1 - lag-1 autocorrelation) times the raw variance, and the
    remainder is attributed to atmospheric turbulence.

    The PBL height for each interval is identified by scanning upward from
    the surface for the first run of n_layers consecutive range gates whose
    atmospheric variance exceeds threshold; the PBL height is reported as the
    height of the n-th (last) gate in that run.

    Note:
    This is a simplified implementation of the turbulence component of the
    Tucker et al. (2009) method and does not incorporate the shear or
    aerosol backscatter components described in the original paper.

    References
    ----------
    Tucker, S. C., et al. (2009), Doppler Lidar Estimation of Mixing Height
    Using Turbulence, Shear, and Aerosol Backscatter Data, J. Atmos. Oceanic
    Technol., 26, 673-688.

    Newsom, RK, and Krishnamurthy, Raglavendra. Doppler Lidar (DL) Instrument Handbook.
    United States: N. p., 2022. Web. doi:10.2172/1034640.

    Jackson, R., O’Brien, J., Wang, J., Fytanidis, D., Muradyan, P.,
    Grover, M., Raut, B., Collis, S., Tuftedal, M., Anderson, G., Wagner, T. J.,
    Nesbitt, S., Tan. H., Wefer, D., & Hammond, M. (2025), The thermodynamic
    and kinematic structure of the planetary boundary
    layer for a summer lake breeze day in Chicago. Journal of Geophysical
    Research: Atmospheres, accepted.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the radial velocity variable.
    velocity : str
        Name of the radial (vertical) velocity variable in ds. Units should be m/s.
    dis_parm : str
        Name of the height/range coordinate in ds.
    interval : str
        Averaging interval, as a pandas offset alias (e.g. "10min"), over
        which the velocity variance and autocorrelation are computed.
    threshold : float
        Atmospheric (turbulent) variance threshold, in the same units as
        velocity squared, above which a range gate is considered part of
        the turbulently mixed layer.
    noise_variance_threshold : float
        Instrument noise variance threshold, in the same units as velocity squared, above which a range gate is considered to have sufficient
        signal-to-noise ratio to be included in the PBL height determination.
        The default value of 0.2 is based on the typical noise characteristics of the ARM Doppler lidars, but may need to be adjusted for other instruments.
    min_gate_height : float
        Minimum height of the range gate to be considered for PBL height determination.
        This is to avoid surface noise and spurious low-level signals. The default value is 100 meters.
        Original dataset with the following variables added:
        pbl_tucker : PBL height for each averaging interval.
        tucker_atmospheric_variance : Atmospheric variance profile for each
            averaging interval and range gate.
        tucker_noise_variance : Instrument noise variance profile for each
            averaging interval and range gate.

    """
    if acf is None:
        raise ImportError("statsmodels is required for the Tucker method but is not installed.")

    vel = ds[velocity]
    height = ds[dis_parm].values
    n_heights = height.size

    interval_times = []
    pbl_heights = []
    noise_variance = []
    atmos_variance = []

    for interval_start, velocity_interval in vel.resample(time=interval):
        vel_interval = velocity_interval.dropna(dim="time", how="all")

        noise_var = np.full(n_heights, np.nan)
        atmos_var = np.full(n_heights, np.nan)

        for i in range(n_heights):
            series = vel_interval.isel({dis_parm: i}).dropna(dim="time").values
            if series.size < 3:
                continue

            total_var = np.var(series)
            acf_values = acf(series, nlags=2, fft=True)
            noise_var[i] = (acf_values[0] - acf_values[1]) * total_var
            atmos_var[i] = total_var - noise_var[i]

        # PBL height is the height of the n-th consecutive range gate,
        # counted from the surface, whose atmospheric variance exceeds
        # threshold
        mask = (atmos_var < threshold) & (noise_var < noise_variance_threshold)
        height_inds = np.argwhere(height > min_gate_height).astype(int).flatten()
        mask = mask[height_inds[0] :]
        match_inds = np.argwhere(mask).flatten()
        if match_inds.size > 0:
            pbl_height = height[height_inds][match_inds[0]]
        else:
            pbl_height = np.nan
        interval_times.append(interval_start + pd.Timedelta(interval) / 2)
        pbl_heights.append(pbl_height)
        noise_variance.append(noise_var)
        atmos_variance.append(atmos_var)

    ds = ds.assign_coords(pbl_time=("pbl_time", interval_times))
    ds = ds.assign(
        pbl_tucker=xr.DataArray(pbl_heights, dims="pbl_time"),
        tucker_noise_variance=xr.DataArray(noise_variance, dims=("pbl_time", dis_parm)),
        tucker_atmospheric_variance=xr.DataArray(atmos_variance, dims=("pbl_time", dis_parm)),
    )

    ds["pbl_tucker"].attrs[
        "description"
    ] = "Planetary Boundary Layer Estimate via the Tucker et al. (2009) turbulence method"
    ds["pbl_tucker"].attrs["input_parameter"] = velocity
    ds["pbl_tucker"].attrs["variance_threshold"] = threshold

    if hasattr(ds[dis_parm], "units"):
        ds["pbl_tucker"].attrs["units"] = ds[dis_parm].attrs["units"]
    else:
        ds["pbl_tucker"].attrs["units"] = "meters"

    ds["tucker_noise_variance"].attrs["description"] = (
        "Instrument noise velocity variance estimated from the lag-1 "
        "autocorrelation of radial velocity"
    )
    ds["tucker_noise_variance"].attrs["units"] = 'm^2/s^2'
    ds["tucker_atmospheric_variance"].attrs["description"] = (
        "Atmospheric (turbulent) velocity variance estimated from the lag-1 "
        "autocorrelation of radial velocity"
    )
    ds["tucker_atmospheric_variance"].attrs["units"] = 'm^2/s^2'

    return ds
