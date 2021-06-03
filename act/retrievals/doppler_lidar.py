"""
Functions for doppler lidar specific retrievals

"""
import numpy as np
import warnings
import xarray as xr


def compute_winds_from_ppi(obj, elevation_name='elevation',
                           azimuth_name='azimuth',
                           radial_velocity_name='radial_velocity',
                           snr_name='signal_to_noise_ratio',
                           intensity_name=None,
                           snr_threshold=0.008, remove_all_missing=False,
                           condition_limit=1.0e4, return_obj=None):
    """
    This function will convert a Doppler Lidar PPI scan into vertical
    distribution of horizontal wind direction and speed.

    Code was adapted by Kenneth E Kehoe from code developed by Rob K
    Newsom. Please see the reference noted below and cite accordingly.

    Parameters
    ----------
    obj : Xarray Dataset Object
        The Dataset object containing PPI scan to be converte into winds.
    elevation_name : str
        The name of the elevation variable in the Dataset object.
    azimuth_name : str
        The name of the azimuth variable in the Dataset object.
    radial_velocity_name : str
        The name of the radial velocity variable in the Dataset object.
    snr_name : str
        The name of the signal to noise variable in the Dataset object.
    intensity_name : str
        The name of the intensity variable in the Dataset object. If this
        is set will use intensity instead of signal to noise ratio.
        variable.
    snr_threshold : float
        The signal to noise lower threshold used to decide which values to use.
    remove_all_missing : boolean
        Option to not add a time step in the returned object where all values
        are set to NaN
    condition_limit : float
        Upper limit used with Normalized data to check if data should be
        converted from scan signal to noise ration to wind speeds and
        directions.
    return_obj : None or  Xarray Dataset Object
        If set to a Xarray Dataset Object the calculated winds object will
        be concatinated onto this object. This is to allow looping over
        this function for many scans and returning a single object.

    Returns
    -------
    obj : Xarray Dataset Object or None
        The winds converted from PPI scan to horizontal wind speeds and wind
        directions along with wind speed error and wind direction error. If
        there is a problem determining the breaks between PPI scans, will
        return None.

    References
    ----------
    Rob K Newsom, Alan W Brewer, James M Wilczak, Daniel E Wolfe,
    Steven P Oncley and Julie K Lundquist ; Validating Precision Estimates in
    Horizontal Wind Measurements from a Doppler Lidar, Atmospheric Measurement
    Techniques Discussions 2016, 10, 1-30

    """
    azimuth = obj[azimuth_name].values
    azimuth_rounded = np.round(azimuth).astype(int)

    # Determine where the azimuth scans repeate to get range for each PPI
    index = np.where(azimuth_rounded == azimuth_rounded[0])[0]
    if index.size == 0:
        print('\nERROR: Having trouble determining the PPI scan breaks '
              'in compute_winds_from_ppi().\n')
        return return_obj

    if index.size == 1:
        num_scans = azimuth.size
    else:
        num_scans = index[1] - index[0]

    # Loop over each PPI scan
    for start_index in index:
        scan_index = range(start_index, start_index + num_scans)
        # Since this can run while instrument is making measurements
        # the number of PPI scans may not match exactly. This will
        # adjust the number of scans in case there is an issue.
        if scan_index[-1] > obj[elevation_name].values.size:
            scan_index = range(start_index, obj[elevation_name].values.size)

        elevation = np.radians(obj[elevation_name].values[scan_index])
        azimuth = np.radians(obj[azimuth_name].values[scan_index])
        doppler = obj[radial_velocity_name].values[scan_index]
        if intensity_name is not None:
            intensity = obj[intensity_name].values[scan_index, :]
            snr = intensity - 1
            del intensity
            snr_name = intensity_name
        else:
            try:
                snr = obj[snr_name].values[scan_index, :]
            except KeyError:
                intensity = obj['intensity'].values[scan_index, :]
                snr = intensity - 1
                del intensity
                snr_name = 'intensity'

        height_name = list(set(obj[snr_name].dims) - set(['time']))[0]
        rng = obj[height_name].values
        try:
            height_units = obj[height_name].attrs['units']
        except KeyError:
            if rng[0] > 0:
                height_units = 'm'
            else:
                height_units = 'km'
        time = obj['time'].values[scan_index]

        height = rng * np.median(np.sin(elevation))
        xhat = np.sin(azimuth) * np.cos(elevation)
        yhat = np.cos(azimuth) * np.cos(elevation)
        zhat = np.sin(elevation)

        dims = snr.shape

        # mean_snr = np.nanmean(snr, axis=1)
        u_wind = np.full(dims[1], np.nan)
        v_wind = np.full(dims[1], np.nan)
        w_wind = np.full(dims[1], np.nan)
        u_err = np.full(dims[1], np.nan)
        v_err = np.full(dims[1], np.nan)
        w_err = np.full(dims[1], np.nan)
        residual = np.full(dims[1], np.nan)
        chisq = np.full(dims[1], np.nan)
        corr = np.full(dims[1], np.nan)

        # Loop over each level
        for ii in range(dims[1]):
            ur1 = doppler[:, ii]
            snr1 = snr[:, ii]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                index = np.where((snr1 >= snr_threshold) & np.isfinite(ur1))[0]
            count = index.size
            if count >= 4:
                ur1 = ur1[index]
                xhat1 = xhat[index]
                yhat1 = yhat[index]
                zhat1 = zhat[index]

                a = np.full((3, 3), np.nan)
                b = np.full(3, np.nan)

                a[0, 0] = np.sum(xhat1**2)
                a[1, 0] = np.sum(xhat1 * yhat1)
                a[2, 0] = np.sum(xhat1 * zhat1)

                a[0, 1] = a[1, 0]
                a[1, 1] = np.sum(yhat1**2)
                a[2, 1] = np.sum(yhat1 * zhat1)

                a[0, 2] = a[2, 0]
                a[1, 2] = a[2, 1]
                a[2, 2] = np.sum(zhat1**2)

                b[0] = np.sum(ur1 * xhat1)
                b[1] = np.sum(ur1 * yhat1)
                b[2] = np.sum(ur1 * zhat1)

                ainv = np.linalg.inv(a)
                condition = np.linalg.norm(a) * np.linalg.norm(ainv)  # Condition Number ?
                if condition < condition_limit:
                    c = b @ ainv
                    u_wind[ii] = c[0]
                    v_wind[ii] = c[1]
                    w_wind[ii] = c[2]
                    ur_fit = xhat1 * u_wind[ii] + yhat1 * v_wind[ii] + zhat1 * w_wind[ii]
                    chisq[ii] = np.sum((ur_fit - ur1)**2)
                    residual[ii] = np.sqrt(chisq[ii] / count)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        corr[ii] = np.corrcoef(ur_fit, ur1)[0, 1]
                    u_err[ii] = np.sqrt((chisq[ii] / (count - 3)) * ainv[0, 0])
                    v_err[ii] = np.sqrt((chisq[ii] / (count - 3)) * ainv[1, 1])
                    w_err[ii] = np.sqrt((chisq[ii] / (count - 3)) * ainv[2, 2])

        # Compute windspeed and direction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            wspd = np.sqrt(u_wind**2 + v_wind**2)
            wdir = np.degrees(np.arctan2(u_wind, v_wind) + np.pi)

            wspd_err = np.sqrt((u_wind * u_err)**2 + (v_wind * v_err)**2) / wspd
            wdir_err = np.degrees(np.sqrt((u_wind * v_err)**2 + (v_wind * u_err)**2) / wspd**2)

        if remove_all_missing and np.isnan(wspd).all():
            continue

        time = time[0] + (time[-1] - time[0]) / 2
        time = time.reshape(1,)
        wspd = wspd.reshape(1, rng.size)
        wdir = wdir.reshape(1, rng.size)
        wspd_err = wspd_err.reshape(1, rng.size)
        wdir_err = wdir_err.reshape(1, rng.size)
        corr = corr.reshape(1, rng.size)
        residual = residual.reshape(1, rng.size)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            snr_mean = np.nanmean(snr, axis=0)
        snr_mean = snr_mean.reshape(1, rng.size)
        new_object = xr.Dataset(
            {'wind_speed': (('time', 'height'), wspd, {'long_name': 'Wind speed', 'units': 'm/s'}),
             'wind_direction': (('time', 'height'), wdir,
                                {'long_name': 'Wind direction', 'units': 'degree'}),
             'wind_speed_error': (('time', 'height'), wspd_err,
                                  {'long_name': 'Wind direction error', 'units': 'm/s'}),
             'wind_direction_error': (('time', 'height'), wdir_err,
                                      {'long_name': 'Wind direction error', 'units': 'degree'}),
             'signal_to_noise_ratio': (('time', 'height'), snr_mean,
                                       {'long_name': 'Signal to noise ratio mean over PPI scan',
                                        'units': '1'}),
             'residual': (('time', 'height'), residual,
                          {'long_name': 'Residual values (Square Root of Chi Square)',
                           'units': 'm/s'}),
             'correlation': (('time', 'height'), corr,
                             {'long_name': 'Correlation coefficient', 'units': '1'})},
            {'time': ('time', time, {'long_name': 'Time in UTC'}),
             'height': ('height', height, {'long_name': 'Height to center of bin',
                                           'units': height_units})},
        )

        if isinstance(return_obj, xr.core.dataset.Dataset):
            return_obj = xr.concat([return_obj, new_object], dim='time')
        else:
            return_obj = new_object

    return return_obj
