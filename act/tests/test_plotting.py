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

from act.plotting import TimeSeriesDisplay
from botocore.handlers import disable_signing

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


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_tuple():
    conn = boto3.resource('s3')
    conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    bucket = conn.Bucket('act-tests')
    for item in bucket.objects.all():
        bucket.download_file(item.key, ('data/' + item.key))

    ceil_ds = arm.read_netcdf('data/sgpceilC1.b1*')
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE_WILDCARD)
    ceil_ds = ceil.correct_ceil(ceil_ds, -9999.)

    # You can use tuples if the datasets in the tuple contain a
    # datastream attribute. This is required in all ARM datasets.
    display = TimeSeriesDisplay(
        (ceil_ds, sonde_ds), subplot_shape=(2,), figsize=(15, 10))
    display.plot('backscatter', 'sgpceilC1.b1', subplot_index=(0,))
    display.plot('temp_mean', 'sgpmetE13.b1', subplot_index=(1,))
    display.day_night_background('sgpmetE13.b1', subplot_index=(1,))
    plt.show()
    return display.fig

@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_dict():
    conn = boto3.resource('s3')
    conn.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)
    bucket = conn.Bucket('act-tests')
    for item in bucket.objects.all():
        bucket.download_file(item.key, ('data/' + item.key))

    ceil_ds = arm.read_netcdf('data/sgpceilC1.b1*')
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE_WILDCARD)
    ceil_ds = ceil.correct_ceil(ceil_ds, -9999.)

    display = TimeSeriesDisplay(
        {'ceiliometer': ceil_ds, 'rawinsonde': sonde_ds},
        subplot_shape=(2,), figsize=(15, 10))
    display.plot('backscatter', 'ceiliometer', subplot_index=(0,))
    display.plot('temp_mean', 'rawinsonde', subplot_index=(1,))
    display.day_night_background('rawinsonde', subplot_index=(1,))
    plt.show()
    return display.fig


