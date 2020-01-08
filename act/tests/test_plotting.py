import act.io.armfiles as arm
import act.tests.sample_files as sample_files
import act.corrections.ceil as ceil
import pytest
import os
import boto3
import numpy as np
import glob
import xarray as xr

from act.plotting import TimeSeriesDisplay, WindRoseDisplay
from act.plotting import SkewTDisplay, XSectionDisplay
from act.plotting import GeographicPlotDisplay, HistogramDisplay
from act.plotting import ContourDisplay
from act.utils.data_utils import accumulate_precip
from botocore.handlers import disable_signing
import matplotlib
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

    windrose = WindRoseDisplay(met)
    display.put_display_in_subplot(windrose, subplot_index=(1, 1))
    windrose.plot('wdir_vec_mean', 'wspd_vec_mean',
                  spd_bins=np.linspace(0, 10, 4))
    windrose.axes[0].legend(loc='best')
    met.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_tuple():
    conn = boto3.resource('s3')
    conn.meta.client.meta.events.register('choose-signer.s3.*',
                                          disable_signing)
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
    ceil_ds.close()
    sonde_ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_dict():
    conn = boto3.resource('s3')
    conn.meta.client.meta.events.register('choose-signer.s3.*',
                                          disable_signing)
    bucket = conn.Bucket('act-tests')
    if not os.path.isdir((os.getcwd() + '/data/')):
        os.makedirs((os.getcwd() + '/data/'))

    for item in bucket.objects.all():
        bucket.download_file(item.key, (os.getcwd() + '/data/' + item.key))

    ceil_ds = arm.read_netcdf('data/sgpceilC1.b1*')
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_MET_WILDCARD)
    ceil_ds = ceil.correct_ceil(ceil_ds, fill_value=-9999.)

    display = TimeSeriesDisplay(
        {'ceiliometer': ceil_ds, 'rawinsonde': sonde_ds},
        subplot_shape=(2,), figsize=(15, 10))
    display.plot('backscatter', 'ceiliometer', subplot_index=(0,))
    display.plot('temp_mean', 'rawinsonde', subplot_index=(1,))
    display.day_night_background('rawinsonde', subplot_index=(1,))
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


@pytest.mark.mpl_image_compare(tolerance=30)
def test_skewt_plot():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    skewt = SkewTDisplay(sonde_ds)

    skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
    sonde_ds.close()

    return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_skewt_plot_spd_dir():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    skewt = SkewTDisplay(sonde_ds)
    skewt.plot_from_spd_and_dir('wspd', 'deg', 'pres', 'tdry', 'dp')
    sonde_ds.close()

    return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_xsection_plot():
    visst_ds = arm.read_netcdf(
        sample_files.EXAMPLE_CEIL1)

    xsection = XSectionDisplay(visst_ds, figsize=(10, 8))
    xsection.plot_xsection(None, 'backscatter', x='time', y='range',
                           cmap='coolwarm', vmin=0, vmax=320)
    visst_ds.close()
    return xsection.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_xsection_plot_map():
    radar_ds = arm.read_netcdf(
        sample_files.EXAMPLE_VISST, combine='nested')
    xsection = XSectionDisplay(radar_ds, figsize=(15, 8))
    xsection.plot_xsection_map(None, 'ir_temperature', vmin=220, vmax=300, cmap='Greys',
                               x='longitude', y='latitude', isel_kwargs={'time': 0})
    radar_ds.close()
    return xsection.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_geoplot():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    geodisplay = GeographicPlotDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    geodisplay.geoplot('tdry', marker='.')
    sonde_ds.close()

    return geodisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stair_graph():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    histdisplay = HistogramDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stairstep_graph('tdry', bins=np.arange(-60, 10, 1))
    sonde_ds.close()

    return histdisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stair_graph_sorted():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    histdisplay = HistogramDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stairstep_graph(
        'tdry', bins=np.arange(-60, 10, 1), sortby_field="alt",
        sortby_bins=np.linspace(0, 10000., 6))
    sonde_ds.close()

    return histdisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stacked_bar_graph():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    histdisplay = HistogramDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar_graph('tdry', bins=np.arange(-60, 10, 1))
    sonde_ds.close()

    return histdisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stacked_bar_graph_sorted():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    histdisplay = HistogramDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar_graph(
        'tdry', bins=np.arange(-60, 10, 1), sortby_field="alt",
        sortby_bins=np.linspace(0, 10000., 6))
    sonde_ds.close()

    return histdisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_heatmap():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    histdisplay = HistogramDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_heatmap(
        'tdry', 'alt', x_bins=np.arange(-60, 10, 1),
        y_bins=np.linspace(0, 10000., 50), cmap='coolwarm')
    sonde_ds.close()

    return histdisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_size_distribution():
    sigma = 10
    mu = 50
    bins = np.linspace(0, 100, 50)
    ydata = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2))
    y_array = xr.DataArray(ydata, dims={'bins': bins})
    bins = xr.DataArray(bins, dims={'bins': bins})
    my_fake_ds = xr.Dataset({'bins': bins, 'ydata': y_array})
    histdisplay = HistogramDisplay(my_fake_ds)
    histdisplay.plot_size_distribution('ydata', 'bins', set_title='Fake distribution.')
    return histdisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contour():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    for f in files:
        obj = arm.read_netcdf(f)
        data.update({f: obj})
        fields.update({f: ['lon', 'lat', 'temp_mean']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50)

    return display.fig


# Due to issues with pytest-mpl, for now we just test to see if it runs
def test_time_height_scatter():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    display = TimeSeriesDisplay({'sgpsondewnpnC1.b1': sonde_ds},
                                figsize=(7, 3))
    display.time_height_scatter('tdry', day_night_background=False)
    sonde_ds.close()

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_qc_bar_plot():
    ds_object = arm.read_netcdf(sample_files.EXAMPLE_MET1)
    ds_object.clean.cleanup()
    var_name = 'temp_mean'
    ds_object.qcfilter.set_test(var_name, index=range(100, 600), test_number=2)

    display = TimeSeriesDisplay({'sgpmetE13.b1': ds_object},
                                subplot_shape=(2, ), figsize=(7, 4))
    display.plot(var_name, subplot_index=(0, ), assessment_overplot=True)
    display.day_night_background('sgpmetE13.b1', subplot_index=(0, ))
    display.qc_flag_block_plot(var_name, subplot_index=(1, ))

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_2d_as_1d():
    obj = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)

    display = TimeSeriesDisplay(obj)
    display.plot('backscatter', force_line_plot=True)

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_fill_between():
    obj = arm.read_netcdf(sample_files.EXAMPLE_MET_WILDCARD)

    accumulate_precip(obj, 'tbrg_precip_total')

    display = TimeSeriesDisplay(obj)
    display.fill_between('tbrg_precip_total_accumulated', color='gray', alpha=0.2)

    return display.fig
