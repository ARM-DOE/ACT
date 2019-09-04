import act.io.armfiles as arm
import act.tests.sample_files as sample_files
#import pytest
#import matplotlib.pyplot as plt
import os
import numpy as np
import glob

from act.plotting import TimeSeriesDisplay
import matplotlib

from act.plotting import _act_cmap, act_cmap

matplotlib.use('Agg')


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot():
    # Process MET data to get simple LCL
    files = sample_files.EXAMPLE_MET_WILDCARD
    met = arm.read_netcdf(files)
    met_temp = met.temp_mean
    met_rh = met.rh_mean
    met_lcl = (20. + met_temp / 5.) * (100. - met_rh) / 1000.
    met['met_lcl'] = met_lcl * 1000.
    met['met_lcl'].attrs['units'] = 'm'
    met['met_lcl'].attrs['long_name'] = 'LCL Calculated from SGP MET E13'

    # Plot data
    display = TimeSeriesDisplay(met)
    display.add_subplots((2, 2), figsize=(15, 10))
    display.plot('wspd_vec_mean', subplot_index=(0, 0))
    display.plot('temp_mean', subplot_index=(1, 0))
    display.plot('rh_mean', subplot_index=(0, 1))

def test_colormaps_exist():
    assert isinstance(act_cmap.HomeyerRainbow, matplotlib.colors.Colormap)
    assert isinstance(act_cmap.HomeyerRainbow, matplotlib.colors.Colormap)


def test_colormaps_registered():
    cmap = matplotlib.cm.get_cmap('act_HomeyerRainbow')
    assert isinstance(cmap, matplotlib.colors.Colormap)