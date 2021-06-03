"""
Tests specific to radiometers.

"""

from scipy.fftpack import rfft, rfftfreq
import numpy as np
import xarray as xr
import pandas as pd
import datetime
import dask
import warnings

from act.utils.datetime_utils import determine_time_delta
from act.utils.geo_utils import get_sunrise_sunset_noon, is_sun_visible


def fft_shading_test(obj, variable='diffuse_hemisp_narrowband_filter4',
                     fft_window=30,
                     shad_freq_lower=[0.008, 0.017],
                     shad_freq_upper=[0.0105, 0.0195],
                     ratio_thresh=[3.15, 1.2],
                     time_interval=None, smooth_window=5, shading_thresh=0.4):
    """
    Function to test shadowband radiometer (MFRSR, RSS, etc) instruments
    for shading related problems.  Program was adapted by Adam Theisen
    from the method defined in Alexandrov et al 2007 to process on a
    point by point basis using a window of data around that point for
    the FFT analysis.

    For ARM data, testing has found that this works the best on narrowband
    filter4 for MFRSR data.

    Function has been tested and is in use by the ARM DQ Office for
    problem detection.  It is know to have some false positives at times.

    Need to run obj.clean.cleanup() ahead of time to ensure proper addition
    to QC variable

    Parameters
    ----------
    obj : xarray Dataset
        Data object
    variable : string
        Name of variable to process
    fft_window : int
        Number of samples to use in the FFT window.  Default is +- 30 samples
        Note: this is +- so the full window will be double
    shad_freq_lower : list
        Lower frequency over which to look for peaks in FFT
    shad_freq_upper : list
        Upper frequency over which to look for peaks in FFT
    ratio_thresh : list
        Threshold for each freq window to flag data.  I.e. if the peak is 3.15 times
        greater than the surrounding area
    time_interval : float
        Sampling rate of the instrument
    smooth_window : int
        Number of samples to use in smoothing FFTs before analysis
    shading_thresh : float
        After smoothing, the value over which is considered a shading signal

    Returns
    -------
    obj : xarray Dataset
        Data object

    References
    ----------
    Alexandrov, Mikhail & Kiedron, Peter & Michalsky, Joseph & Hodges, Gary
    & Flynn, Connor & Lacis, Andrew. (2007). Optical depth measurements by
    shadow-band radiometers and their uncertainties. Applied optics. 46.
    8027-38. 10.1364/AO.46.008027.

    """

    # Get time and data from variable
    time = obj['time'].values
    data = obj[variable].values
    if 'missing_value' in obj[variable].attrs:
        missing = obj[variable].attrs['missing_value']
    else:
        missing = -9999.

    # Get time interval between measurements
    if time_interval is None:
        dt = determine_time_delta(time)
    else:
        dt = time_interval

    # Compute the FFT for each point +- window samples
    task = []
    sun_up = is_sun_visible(latitude=obj['lat'].values, longitude=obj['lon'].values, date_time=time)
    for t in range(len(time)):
        sind = t - fft_window
        eind = t + fft_window
        if sind < 0:
            sind = 0
        if eind > len(time):
            eind = len(time)

        # Get data and remove all nan/missing values
        d = data[sind:eind]
        idx = ((d != missing) & (np.isnan(d) is not True))
        index = np.where(idx)
        d = d[index]

        # Add to task for dask processing
        task.append(dask.delayed(fft_shading_test_process)(
            time[t], d,
            shad_freq_lower=shad_freq_lower,
            shad_freq_upper=shad_freq_upper,
            ratio_thresh=ratio_thresh,
            time_interval=dt,
            is_sunny=sun_up[t]))

    # Process using dask
    result = dask.compute(*task)

    # Run data through a rolling median to filter out singular
    # false positives
    shading = [r['shading'] for r in result]
    shading = pd.Series(shading).rolling(window=smooth_window, min_periods=1).median()

    # Find indices where shading is indicated
    idx = (np.asarray(shading) > shading_thresh)
    index = np.where(idx)

    # Add test to QC Variable
    desc = 'FFT Shading Test'
    obj.qcfilter.add_test(variable, index=index, test_meaning=desc)

    # Prepare frequency and fft variables for adding to object
    fft = np.empty([len(time), fft_window * 2])
    fft[:] = np.nan
    freq = np.empty([len(time), fft_window * 2])
    freq[:] = np.nan
    for i, r in enumerate(result):
        dummy = r['fft']
        fft[i, 0:len(dummy)] = dummy
        dummy = r['freq']
        freq[i, 0:len(dummy)] = dummy

    attrs = {'units': '', 'long_name': 'FFT Results for Shading Test', 'upper_freq': shad_freq_upper,
             'lower_freq': shad_freq_lower}
    fft_window = xr.DataArray(range(fft_window * 2), dims=['fft_window'],
                              attrs={'long_name': 'FFT Window', 'units': '1'})
    da = xr.DataArray(fft, dims=['time', 'fft_window'], attrs=attrs, coords=[obj['time'], fft_window])
    obj['fft'] = da
    attrs = {'units': '', 'long_name': 'FFT Frequency Values for Shading Test'}
    da = xr.DataArray(freq, dims=['time', 'fft_window'], attrs=attrs, coords=[obj['time'], fft_window])
    obj['fft_freq'] = da

    return obj


def fft_shading_test_process(time, data, shad_freq_lower=None,
                             shad_freq_upper=None, ratio_thresh=None,
                             time_interval=None, is_sunny=None):
    """
    Processing function to do the FFT calculations/thresholding

    Parameters
    ----------
    time : datetime
        Center time of calculation used for calculating sunrise/sunset
    data : list
        Data for run through fft processing
    shad_freq_lower : list
        Lower limits of freqencies to look for shading issues
    shad_freq_upper : list
        Upper limits of freqencies to look for shading issues
    ratio_thresh : list
        Thresholds to apply, corresponding to frequencies chosen
    time_interval : float
        Time interval of data

    Returns
    -------
    shading : int
        Binary to indicate shading problem (1) or not (0)

    """

    if not is_sunny:
        return {'shading': 0, 'fft': [np.nan] * len(data), 'freq': [np.nan] * len(data)}

    # FFT Algorithm
    fftv = abs(rfft(data))
    freq = rfftfreq(fftv.size, d=time_interval)

    # Get FFT data under threshold
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        idx = (fftv > 1.)
    index = np.where(idx)
    fftv[index] = np.nan
    freq[index] = np.nan

    # Return if FFT is empty
    if len(fftv) == 0:
        return {'shading': 0, 'fft': [np.nan] * len(data), 'freq': [np.nan] * len(data)}

    # Commented out as it seems to work better without smoothing
    # fftv=pd.DataFrame(data=fftv).rolling(min_periods=3,window=3,center=True).mean().values.flatten()

    ratio = []
    # Calculates the ratio (size) of the peaks in the FFT to the surrounding
    # data
    wind = 3

    # Run through each frequency to look for peaks
    # Calculate threshold of peak value to surrounding values
    for i in range(len(shad_freq_lower)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            idx = np.logical_and(freq > shad_freq_lower[i],
                                 freq < shad_freq_upper[i])

        index = np.where(idx)
        if len(index[0]) == 0:
            continue
        peak = max(fftv[index])
        index = index[0]

        sind = index[0] - wind
        if sind < 0:
            sind = 0
        eind = index[-1] + wind
        if eind > len(fftv):
            eind = len(fftv)

        if len(range(sind, index[0])) == 0 or len(range(index[-1], eind)) == 0:
            ratio.append(0.0)
        else:
            # Calculates to the left/right of each peak
            peak_l = max(fftv[range(sind, index[0])])
            peak_r = max(fftv[range(index[-1], eind)])
            ratio.append(peak / np.mean([peak_l, peak_r]))

    # Checks ratios against thresholds for each freq range
    shading = 0
    if len(ratio) > 0:
        pass1 = False
        pass2 = False
        if ratio[0] > ratio_thresh[0]:
            pass1 = True
        if len(ratio) > 1:
            if ratio[1] > ratio_thresh[1]:
                pass2 = True
        else:
            pass2 = True

        if pass1 and pass2:
            shading = 1

    return {'shading': shading, 'fft': fftv, 'freq': freq}
