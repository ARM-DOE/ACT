import warnings

import numpy as np
import pandas as pd

try:
    import pysp2

    PYSP2_AVAILABLE = True
except ImportError:
    PYSP2_AVAILABLE = False

if PYSP2_AVAILABLE:

    class SP2ParticleCriteria:
        """
        This class stores the particle crtiteria for filtering out bad particles in the SP2.
        In addition, this class stores the calibration statistics for the mass calculations.

        Parameters
        ----------
        cal_file_name: str or None
            Path to the SP2 calibration file. Set to None to use default values.

        Attributes
        ----------
        ScatMaxPeakHt1: int
            The maximum peak value for scattering channel 0.
        ScatMinPeakHt1: int
            The minimum peak value for scattering channel 0
        ScatMaxPeakHt2: int
            The maximum peak value for scattering channel 4.
        ScatMinPeakHt2: int
            The minimum peak value for scattering channel 4
        ScatMinWidth: int
            The minimum peak width for the scattering channels
        ScatMaxWidth: int
            The maximum peak width for the scattering channels
        IncanMinPeakHt1: int
            The minimum peak height for channel 1
        IncanMinPeakHt2: int
            The minimum peak height for channel 5
        IncanMidWidth: int
            The minimum width for the incandescence channels
        IncanMaxWidth: int
            The maximum width for the incandescence channels
        IncanMinPeakPos: int
            The minimum peak position for the incandescence channels.
        IncanMaxPeakPos: int
            The maximum peak position for the incandescence channels.
        IncanMinPeakRatio: float
            The minimum peak ch5/ch6 peak ratio.
        IncanMaxPeakRatio: float
            The maximum peak ch5/ch6 peak ratio.
        c0Mass1, c1Mass1, c2Mass1: floats
            Calibration mass coefficients for ch1
        c0Mass2, c1Mass2, c2Mass2: floats
            Calibration mass coefficients for ch2
        c0Scat1, c1Scat1, c2Scat1: floats
            Calibration scattering coefficients for ch0
        c0Scat2, c1Scat2, c2Scat2: floats
            Calibration scattering coefficients for ch4
        densitySO4, BC: float
            Density of SO4, BC
        tempSTP, presSTP: float
            Temperature [Kelvin] and pressure [hPa] at STP.

        """

        def __init__(self, cal_file_name=None):
            self.ScatMaxPeakHt1 = 60000
            self.ScatMinPeakHt1 = 250
            self.ScatMaxPeakHt2 = 60000
            self.ScatMinPeakHt2 = 250
            self.ScatMinWidth = 10
            self.ScatMaxWidth = 90
            self.ScatMinPeakPos = 20
            self.ScatMaxPeakPos = 90
            self.IncanMinPeakHt1 = 200
            self.IncanMinPeakHt2 = 200
            self.IncanMaxPeakHt1 = 60000
            self.IncanMaxPeakHt2 = 60000
            self.IncanMinWidth = 5
            self.IncanMaxWidth = np.inf
            self.IncanMinPeakPos = 20
            self.IncanMaxPeakPos = 90
            self.IncanMinPeakRatio = 0.1
            self.IncanMaxPeakRatio = 25
            self.IncanMaxPeakOffset = 11
            self.c0Mass1 = 0
            self.c1Mass1 = 0.0001896
            self.c2Mass1 = 0
            self.c3Mass1 = 0
            self.c0Mass2 = 0
            self.c1Mass2 = 0.0016815
            self.c2Mass2 = 0
            self.c3Mass2 = 0
            self.c0Scat1 = 0
            self.c1Scat1 = 78.141
            self.c2Scat1 = 0
            self.c0Scat2 = 0
            self.c1Scat2 = 752.53
            self.c2Scat2 = 0
            self.densitySO4 = 1.8
            self.densityBC = 1.8
            self.TempSTP = 273.15
            self.PressSTP = 1013.25
            if cal_file_name is not None:
                df = pd.read_csv(cal_file_name, sep='\t')
                for i in range(len(df['CalName'].values)):
                    setattr(self, df['CalName'].values[i], df['CalValue'].values[i])
                del df

else:

    class SP2ParticleCriteria:
        def __init__(self):
            warnings.warn(
                'Attempting to use SP2ParticleCriteria without'
                'PySP2 installed. SP2ParticleCriteria will'
                'not have any functionality besides this'
                'warning message.',
                RuntimeWarning,
            )


def get_waveform_statistics(ds, config_file, parallel=False, num_records=None):
    """
    Generates waveform statistics for each particle in the dataset
    This will do the fitting for channel 0 only.

    Parameters
    ----------
    ds : xarray.Dataset
        Raw SP2 binary dataset
    config_file: ConfigParser object
        The configuration INI file path.
    parallel: bool
        If true, use dask to enable parallelism
    num_records: int or None
        Only process first num_records datapoints. Set to
        None to process all records.

    Returns
    -------
    wave_ds: xarray.Dataset
        Dataset with gaussian fits
    """
    if PYSP2_AVAILABLE:
        config = pysp2.io.read_config(config_file)
        return pysp2.util.gaussian_fit(ds, config, parallel, num_records)
    else:
        raise ModuleNotFoundError('PySP2 must be installed in order to process SP2 data.')
