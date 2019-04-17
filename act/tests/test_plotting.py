import matplotlib
matplotlib.use('Agg')
import act.io.armfiles as arm
import act.discovery.get_files as get_data
import act.tests.sample_files as sample_files
import act.corrections.ceil as ceil
import pytest
import glob
import matplotlib.pyplot as plt
import os
import boto3
import numpy as np

from act.plotting import TimeSeriesDisplay, WindRoseDisplay
from botocore.handlers import disable_signing


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
    # Plot data
    display = TimeSeriesDisplay(met)
    display.add_subplots((3,), figsize=(15, 10))
    display.plot('wspd_vec_mean', subplot_index=(0, ))
    display.plot('temp_mean', subplot_index=(1, ))
    display.plot('rh_mean', subplot_index=(2, ))
    met.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_tuple():
    conn = boto3.resource('s3')
    conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    bucket = conn.Bucket('act-tests')
    if not os.path.isdir((os.getcwd() + '/data/')):
        os.makedirs((os.getcwd() + '/data/'))

    for item in bucket.objects.all():
        bucket.download_file(item.key, (os.getcwd() + '/data/' + item.key))

    ceil_ds = arm.read_netcdf('data/sgpceilC1.b1*')
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_MET_WILDCARD)
    # Removing fill value of -9999 as it was causing some warnings
    ceil_ds = ceil.correct_ceil(ceil_ds)

    # You can use tuples if the datasets in the tuple contain a
    # datastream attribute. This is required in all ARM datasets.
    display = TimeSeriesDisplay(
        (ceil_ds, sonde_ds), subplot_shape=(2,), figsize=(15, 10))
    display.plot('backscatter', 'sgpceilC1.b1', subplot_index=(0,))
    display.plot('temp_mean', 'sgpmetE13.b1', subplot_index=(1,))
    display.day_night_background('sgpmetE13.b1', subplot_index=(1,))
    plt.show()
    ceil_ds.close()
    sonde_ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_dict():
    conn = boto3.resource('s3')
    conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    bucket = conn.Bucket('act-tests')
    if not os.path.isdir((os.getcwd() + '/data/')):
        os.makedirs((os.getcwd() + '/data/'))

    for item in bucket.objects.all():
        bucket.download_file(item.key, (os.getcwd() + '/data/' + item.key))

    ceil_ds = arm.read_netcdf('data/sgpceilC1.b1*')
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_MET_WILDCARD)
    ceil_ds = ceil.correct_ceil(ceil_ds, -9999.)

    display = TimeSeriesDisplay(
        {'ceiliometer': ceil_ds, 'rawinsonde': sonde_ds},
        subplot_shape=(2,), figsize=(15, 10))
    display.plot('backscatter', 'ceiliometer', subplot_index=(0,))
    display.plot('temp_mean', 'rawinsonde', subplot_index=(1,))
    display.day_night_background('rawinsonde', subplot_index=(1,))
    plt.show()
    ceil_ds.close()
    sonde_ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_wind_rose():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_TWP_SONDE_WILDCARD)

    WindDisplay = WindRoseDisplay(sonde_ds, figsize=(10, 10))
    WindDisplay.plot('deg', 'wspd',
                     spd_bins=np.linspace(0, 20, 10), num_dirs=30,
                     tick_interval=2)
    sonde_ds.close()
    return WindDisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_barb_sounding_plot():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
    BarbDisplay = TimeSeriesDisplay({'sonde_darwin': sonde_ds})
    BarbDisplay.plot_time_height_xsection_from_1d_data('rh', 'pres',
                                                       cmap='coolwarm_r',
                                                       vmin=0, vmax=100,
                                                       num_time_periods=25)
    BarbDisplay.plot_barbs_from_spd_dir('deg', 'wspd', 'pres',
                                        num_barbs_x=20)
    sonde_ds.close()
    return BarbDisplay.fig
