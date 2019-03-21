import matplotlib
matplotlib.use('Agg')
import act.io.armfiles as arm
import act.discovery.get_files as get_data
import act.tests.sample_files as sample_files
import pytest
import glob
import matplotlib.pyplot as plt
import os

from act.plotting import TimeSeriesDisplay
@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot():
    # Process MET data to get simple LCL
    files = sample_files.EXAMPLE_SONDE_WILDCARD
    met = arm.read_netcdf(files)
    met_temp = met.temp_mean
    met_rh = met.rh_mean
    met_lcl = (20. + met_temp / 5.) * (100. - met_rh) / 1000.
    met['met_lcl'] = met_lcl * 1000.
    met['met_lcl'].attrs['units'] = 'm'
    met['met_lcl'].attrs['long_name'] = 'LCL Calculated from SGP MET E13'

    # Plot data
    # Plot data
    display = TimeSeriesDisplay(met)
    display.add_subplots((3,), figsize=(15, 10))
    display.plot('wspd_vec_mean', subplot_index=(0, ))
    display.plot('temp_mean', subplot_index=(1, ))
    display.plot('rh_mean', subplot_index=(2, ))

    return display.fig
