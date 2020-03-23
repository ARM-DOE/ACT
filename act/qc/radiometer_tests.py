"""
act.qc.radiometer_tests
------------------------------

Tests specific to radiometers
"""

from scipy.fftpack import rfft, rfftfreq
import numpy as np
import astral
import datetime
import pandas as pd
import dask
from act.utils.datetime_utils import determine_time_delta


def fft_shading_test(obj, variable='diffuse_hemisp_narrowband_filter4',
                     fft_window=30,
                     shad_freq_lower=[0.008, 0.017],
                     shad_freq_upper=[0.0105, 0.0195],
                     ratio_thresh=[3.15, 1.2],
                     time_interval=None):
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
    dt = time_interval
    if time_interval is None:
        dt = determine_time_delta(time)

    # Compute the FFT for each point +- window samples
    task = []
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
        lat = [obj['lat'].values] if not isinstance(obj['lat'].values, list) else obj['lat'].values
        lon = [obj['lon'].values] if not isinstance(obj['lon'].values, list) else obj['lon'].values
        task.append(dask.delayed(fft_shading_test_process)(time[t],
                    lat[0], lon[0], d,
                    shad_freq_lower=shad_freq_lower,
                    shad_freq_upper=shad_freq_upper,
                    ratio_thresh=ratio_thresh,
                    time_interval=dt))

    # Process using dask
    result = dask.compute(*task)

    # Run data through a rolling median to filter out singular
    # false positives
    result = pd.Series(result).rolling(window=5, min_periods=1).median()

    # Find indices where shading is indicated
    idx = (np.asarray(result) > 0.4)
    index = np.where(idx)

    # Add test to QC Variable
    desc = 'FFT Shading Test'
    result = obj.qcfilter.add_test(variable, index=index, test_meaning=desc)

    return obj


def fft_shading_test_process(time, lat, lon, data, shad_freq_lower=None,
                             shad_freq_upper=None, ratio_thresh=None,
                             time_interval=None):
    """
    Processing function to do the FFT calculations/thresholding

    Parameters
    ----------
    time : datetime
        Center time of calculation used for calculating sunrise/sunset
    lat : float
        Latitude used for calculating sunrise/sunset
    lon : float
        Longitude used for calculating sunrise/sunset
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

    # Spin up Astral instance for sunrise/sunset calcs
    # Get sunrise/sunset that are on same days
    # This is used to help in the processing and easily exclude
    # nighttime data from processing
    a = astral.Astral()
    a.solar_depression = 0
    sun = a.sun_utc(pd.Timestamp(time), float(lat), float(lon))
    sr = sun['sunrise'].replace(tzinfo=None)
    ss = sun['sunset'].replace(tzinfo=None)
    delta = ss.date() - sr.date()
    if delta > datetime.timedelta(days=0):
        sun = a.sun_utc(pd.Timestamp(time) - datetime.timedelta(days=1),
                        float(lat), float(lon))
        ss = sun['sunset'].replace(tzinfo=None)

    # Set if night or not
    shading = 0
    night = False
    if sr < ss:
        if (pd.Timestamp(time) < sr) or (pd.Timestamp(time) > ss):
            night = True
    if sr > ss:
        if (pd.Timestamp(time) < sr) and (pd.Timestamp(time) > ss):
            night = True

    # Return shading of 0 if no valid data or it's night
    if len(data) == 0 or night is True:
        return shading

    # FFT Algorithm
    fftv = abs(rfft(data))
    freq = rfftfreq(fftv.size, d=time_interval)

    # Get FFT data under threshold
    idx = (fftv < 1.)
    index = np.where(idx)
    fftv = fftv[index]
    freq = freq[index]

    # Return if FFT is empty
    if len(fftv) == 0:
        return 0
    # Commented out as it seems to work better without smoothing
    # fftv=pd.DataFrame(data=fftv).rolling(min_periods=3,window=3,center=True).mean().values.flatten()

    ratio = []
    # Calculates the ratio (size) of the peaks in the FFT to the surrounding
    # data
    wind = 3

    # Run through each frequency to look for peaks
    # Calculate threshold of peak value to surrounding values
    for i in range(len(shad_freq_lower)):
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

    return shading
