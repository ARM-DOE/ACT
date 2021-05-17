import pytest
import numpy as np
import glob
import xarray as xr
import pandas as pd
import os
import act
import act.io.armfiles as arm
import act.tests.sample_files as sample_files
from act.plotting import TimeSeriesDisplay, WindRoseDisplay
from act.plotting import SkewTDisplay, XSectionDisplay
from act.plotting import GeographicPlotDisplay, HistogramDisplay
from act.plotting import ContourDisplay
from act.utils.data_utils import accumulate_precip
import matplotlib
matplotlib.use('Agg')
try:
    import metpy.calc as mpcalc
    METPY = True
except ImportError:
    METPY = False


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


def test_errors():
    files = sample_files.EXAMPLE_MET_WILDCARD
    obj = arm.read_netcdf(files)

    display = TimeSeriesDisplay(obj)
    display.axes = None
    with np.testing.assert_raises(RuntimeError):
        display.day_night_background()

    display = TimeSeriesDisplay({'met': obj, 'met2': obj})
    with np.testing.assert_raises(ValueError):
        display.plot('temp_mean')
    with np.testing.assert_raises(ValueError):
        display.qc_flag_block_plot('qc_temp_mean')
    with np.testing.assert_raises(ValueError):
        display.plot_barbs_from_spd_dir('wdir_vec_mean', 'wspd_vec_mean')
    with np.testing.assert_raises(ValueError):
        display.plot_barbs_from_u_v('wdir_vec_mean', 'wspd_vec_mean')

    del obj.attrs['_file_dates']

    data = np.empty(len(obj['time'])) * np.nan
    lat = obj['lat'].values
    lon = obj['lon'].values
    obj['lat'].values = data
    obj['lon'].values = data

    display = TimeSeriesDisplay(obj)
    display.plot('temp_mean')
    display.set_yrng([0, 0])
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()
    obj['lat'].values = lat
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()
    obj['lon'].values = lon * 100.
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()
    obj['lat'].values = lat * 100.
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()

    obj.close()

    # Test some of the other errors
    obj = arm.read_netcdf(files)
    del obj['temp_mean'].attrs['units']
    display = TimeSeriesDisplay(obj)
    display.axes = None
    with np.testing.assert_raises(RuntimeError):
        display.set_yrng([0, 10])
    with np.testing.assert_raises(RuntimeError):
        display.set_xrng([0, 10])
    display.fig = None
    display.plot('temp_mean', add_nan=True)

    assert display.fig is not None
    assert display.axes is not None

    with np.testing.assert_raises(AttributeError):
        display = TimeSeriesDisplay([])

    fig, ax = matplotlib.pyplot.subplots()
    display = TimeSeriesDisplay(obj)
    display.add_subplots((2, 2), figsize=(15, 10))
    display.assign_to_figure_axis(fig, ax)
    assert display.fig is not None
    assert display.axes is not None

    obj = arm.read_netcdf(files)
    display = TimeSeriesDisplay(obj)
    obj.clean.cleanup()
    display.axes = None
    display.fig = None
    display.qc_flag_block_plot('atmos_pressure')
    assert display.fig is not None
    assert display.axes is not None


def test_histogram_errors():
    files = sample_files.EXAMPLE_MET1
    obj = arm.read_netcdf(files)

    histdisplay = HistogramDisplay(obj)
    histdisplay.axes = None
    with np.testing.assert_raises(RuntimeError):
        histdisplay.set_yrng([0, 10])
    with np.testing.assert_raises(RuntimeError):
        histdisplay.set_xrng([-40, 40])
    histdisplay.fig = None
    histdisplay.plot_stacked_bar_graph('temp_mean', bins=np.arange(-40, 40, 5))
    histdisplay.set_yrng([0, 0])
    assert histdisplay.yrng[0][1] == 1.
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    with np.testing.assert_raises(AttributeError):
        HistogramDisplay([])

    histdisplay.axes = None
    histdisplay.fig = None
    histdisplay.plot_stairstep_graph('temp_mean', bins=np.arange(-40, 40, 5))
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    sigma = 10
    mu = 50
    bins = np.linspace(0, 100, 50)
    ydata = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu)**2 / (2 * sigma**2))
    y_array = xr.DataArray(ydata, dims={'bins': bins})
    bins = xr.DataArray(bins, dims={'bins': bins})
    my_fake_ds = xr.Dataset({'bins': bins, 'ydata': y_array})
    histdisplay = HistogramDisplay(my_fake_ds)
    histdisplay.axes = None
    histdisplay.fig = None
    histdisplay.plot_size_distribution('ydata', 'bins', set_title='Fake distribution.')
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)
    histdisplay = HistogramDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.axes = None
    histdisplay.fig = None
    histdisplay.plot_heatmap('tdry', 'alt', x_bins=np.arange(-60, 10, 1),
                             y_bins=np.linspace(0, 10000., 50), cmap='coolwarm')
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None


def test_xsection_errors():
    obj = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)

    display = XSectionDisplay(obj, figsize=(10, 8), subplot_shape=(2,))
    display.axes = None
    with np.testing.assert_raises(RuntimeError):
        display.set_yrng([0, 10])
    with np.testing.assert_raises(RuntimeError):
        display.set_xrng([-40, 40])

    display = XSectionDisplay(obj, figsize=(10, 8), subplot_shape=(1,))
    with np.testing.assert_raises(RuntimeError):
        display.plot_xsection(None, 'backscatter', x='time')

    obj.close()


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_tuple():
    obj = arm.read_netcdf(sample_files.EXAMPLE_MET1)
    obj2 = arm.read_netcdf(sample_files.EXAMPLE_SIRS)
    obj = obj.rename({'lat': 'fun_time'})
    obj['fun_time'].attrs['standard_name'] = 'latitude'
    obj = obj.rename({'lon': 'not_so_fun_time'})
    obj['not_so_fun_time'].attrs['standard_name'] = 'longitude'

    # You can use tuples if the datasets in the tuple contain a
    # datastream attribute. This is required in all ARM datasets.
    display = TimeSeriesDisplay(
        (obj, obj2), subplot_shape=(2,), figsize=(15, 10))
    display.plot('short_direct_normal', 'sgpsirsE13.b1', subplot_index=(0,))
    display.day_night_background('sgpsirsE13.b1', subplot_index=(0,))
    display.plot('temp_mean', 'sgpmetE13.b1', subplot_index=(1,))
    display.day_night_background('sgpmetE13.b1', subplot_index=(1,))

    ax = act.plotting.common.parse_ax(ax=None)
    ax, fig = act.plotting.common.parse_ax_fig(ax=None, fig=None)
    obj.close()
    obj2.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_dict():

    obj = arm.read_netcdf(sample_files.EXAMPLE_MET1)
    obj2 = arm.read_netcdf(sample_files.EXAMPLE_SIRS)

    # You can use tuples if the datasets in the tuple contain a
    # datastream attribute. This is required in all ARM datasets.
    display = TimeSeriesDisplay(
        {'sirs': obj2, 'met': obj},
        subplot_shape=(2,), figsize=(15, 10))
    display.plot('short_direct_normal', 'sirs', subplot_index=(0,))
    display.day_night_background('sirs', subplot_index=(0,))
    display.plot('temp_mean', 'met', subplot_index=(1,))
    display.day_night_background('met', subplot_index=(1,))
    obj.close()
    obj2.close()

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_wind_rose():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_TWP_SONDE_WILDCARD)

    WindDisplay = WindRoseDisplay(sonde_ds, figsize=(10, 10))
    WindDisplay.plot('deg', 'wspd',
                     spd_bins=np.linspace(0, 20, 10), num_dirs=30,
                     tick_interval=2, cmap='viridis')
    WindDisplay.set_thetarng(trng=(0., 360.))
    WindDisplay.set_rrng((0., 14))

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

    if METPY:
        skewt = SkewTDisplay(sonde_ds)
        skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
        sonde_ds.close()
        return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_skewt_plot_spd_dir():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    if METPY:
        skewt = SkewTDisplay(sonde_ds, ds_name='act_datastream')
        skewt.plot_from_spd_and_dir('wspd', 'deg', 'pres', 'tdry', 'dp')
        sonde_ds.close()
        return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=67)
def test_multi_skewt_plot():

    files = glob.glob(sample_files.EXAMPLE_TWP_SONDE_20060121)
    test = {}
    for f in files:
        time = f.split('.')[-3]
        sonde_ds = arm.read_netcdf(f)
        test.update({time: sonde_ds})

    if METPY:
        skewt = SkewTDisplay(test, subplot_shape=(2, 2))
        i = 0
        j = 0
        for f in files:
            time = f.split('.')[-3]
            skewt.plot_from_spd_and_dir('wspd', 'deg', 'pres', 'tdry', 'dp',
                                        subplot_index=(j, i), dsname=time,
                                        p_levels_to_plot=np.arange(10., 1000., 25))
            if j == 1:
                i += 1
                j = 0
            elif j == 0:
                j += 1
        return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=31)
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
    try:
        xsection = XSectionDisplay(radar_ds, figsize=(15, 8))
        xsection.plot_xsection_map(None, 'ir_temperature', vmin=220, vmax=300, cmap='Greys',
                                   x='longitude', y='latitude', isel_kwargs={'time': 0})
        radar_ds.close()
        return xsection.fig
    except Exception:
        pass


@pytest.mark.mpl_image_compare(tolerance=30)
def test_geoplot():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)
    try:
        geodisplay = GeographicPlotDisplay({'sgpsondewnpnC1.b1': sonde_ds})
        geodisplay.geoplot('tdry', marker='.', cartopy_feature=['STATES', 'LAND', 'OCEAN', 'COASTLINE',
                                                                'BORDERS', 'LAKES', 'RIVERS'],
                           text={'Ponca City': [-97.0725, 36.7125]})
        return geodisplay.fig
    except Exception:
        pass
    sonde_ds.close()


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
def test_stacked_bar_graph2():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_SONDE1)

    histdisplay = HistogramDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar_graph('tdry')
    histdisplay.set_yrng([0, 400])
    histdisplay.set_xrng([-70, 0])
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
    wind_fields = {}
    station_fields = {}
    for f in files:
        obj = arm.read_netcdf(f)
        data.update({f: obj})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50,
                           contour='contour', cmap='viridis')
    display.plot_vectors_from_spd_dir(fields=wind_fields, time=time, mesh=True, grid_delta=(0.1, 0.1))
    display.plot_station(fields=station_fields, time=time, markersize=7, color='red')

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contour_stamp():
    files = glob.glob(sample_files.EXAMPLE_STAMP_WILDCARD)
    test = {}
    stamp_fields = {}
    time = '2020-01-01T00:00:00.000000000'
    for f in files:
        ds = f.split('/')[-1]
        obj = act.io.armfiles.read_netcdf(f)
        test.update({ds: obj})
        stamp_fields.update({ds: ['lon', 'lat', 'plant_water_availability_east']})
        obj.close()

    display = act.plotting.ContourDisplay(test, figsize=(8, 8))
    display.create_contour(fields=stamp_fields, time=time, levels=50,
                           alpha=0.5, twod_dim_value=5)

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contour2():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        obj = arm.read_netcdf(f)
        data.update({f: obj})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50,
                           contour='contour', cmap='viridis')
    display.plot_vectors_from_spd_dir(fields=wind_fields, time=time, mesh=False, grid_delta=(0.1, 0.1))
    display.plot_station(fields=station_fields, time=time, markersize=7, color='pink')

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contourf():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        obj = arm.read_netcdf(f)
        data.update({f: obj})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure', 'temp_mean', 'rh_mean',
                                   'vapor_pressure_mean', 'temp_std']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50,
                           contour='contourf', cmap='viridis')
    display.plot_vectors_from_spd_dir(fields=wind_fields, time=time, mesh=True, grid_delta=(0.1, 0.1))
    display.plot_station(fields=station_fields, time=time, markersize=7, color='red')

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contourf2():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        obj = arm.read_netcdf(f)
        data.update({f: obj})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure', 'temp_mean', 'rh_mean',
                                   'vapor_pressure_mean', 'temp_std']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50,
                           contour='contourf', cmap='viridis')
    display.plot_vectors_from_spd_dir(fields=wind_fields, time=time, mesh=False, grid_delta=(0.1, 0.1))
    display.plot_station(fields=station_fields, time=time, markersize=7, color='pink')

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

    # Testing out when the assessment is not listed
    ds_object.qcfilter.set_test(var_name, index=range(500, 800), test_number=4)
    ds_object['qc_' + var_name].attrs['flag_assessments'][3] = 'Wonky'

    display = TimeSeriesDisplay({'sgpmetE13.b1': ds_object},
                                subplot_shape=(2, ), figsize=(7, 4))
    display.plot(var_name, subplot_index=(0, ), assessment_overplot=True)
    display.day_night_background('sgpmetE13.b1', subplot_index=(0, ))
    color_lookup = {'Bad': 'red', 'Incorrect': 'red',
                    'Indeterminate': 'orange', 'Suspect': 'orange',
                    'Missing': 'darkgray', 'Not Failing': 'green',
                    'Acceptable': 'green'}
    display.qc_flag_block_plot(var_name, subplot_index=(1, ),
                               assessment_color=color_lookup)

    ds_object.close()

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_2d_as_1d():
    obj = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)

    display = TimeSeriesDisplay(obj)
    display.plot('backscatter', force_line_plot=True)

    obj.close()
    del obj

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_fill_between():
    obj = arm.read_netcdf(sample_files.EXAMPLE_MET_WILDCARD)

    accumulate_precip(obj, 'tbrg_precip_total')

    display = TimeSeriesDisplay(obj)
    display.fill_between('tbrg_precip_total_accumulated', color='gray', alpha=0.2)

    obj.close()
    del obj

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_qc_flag_block_plot():
    obj = arm.read_netcdf(sample_files.EXAMPLE_SURFSPECALB1MLAWER)

    display = TimeSeriesDisplay(obj, subplot_shape=(2, ), figsize=(8, 2 * 4))

    display.plot('surface_albedo_mfr_narrowband_10m', force_line_plot=True, labels=True)

    display.qc_flag_block_plot('surface_albedo_mfr_narrowband_10m', subplot_index=(1, ))

    obj.close()
    del obj

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_assessment_overplot():
    var_name = 'temp_mean'
    files = sample_files.EXAMPLE_MET1
    ds = arm.read_netcdf(files)
    ds.load()
    ds.clean.cleanup()

    ds.qcfilter.set_test(var_name, index=np.arange(100, 300, dtype=int), test_number=2)
    ds.qcfilter.set_test(var_name, index=np.arange(420, 422, dtype=int), test_number=3)
    ds.qcfilter.set_test(var_name, index=np.arange(500, 800, dtype=int), test_number=4)
    ds.qcfilter.set_test(var_name, index=np.arange(900, 901, dtype=int), test_number=4)

    # Plot data
    display = TimeSeriesDisplay(ds, subplot_shape=(1, ), figsize=(10, 6))
    display.plot(var_name, day_night_background=True, assessment_overplot=True)

    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_assessment_overplot_multi():
    var_name1, var_name2 = 'wspd_arith_mean', 'wspd_vec_mean'
    files = sample_files.EXAMPLE_MET1
    ds = arm.read_netcdf(files)
    ds.load()
    ds.clean.cleanup()

    ds.qcfilter.set_test(var_name1, index=np.arange(100, 200, dtype=int), test_number=2)
    ds.qcfilter.set_test(var_name1, index=np.arange(500, 600, dtype=int), test_number=4)
    ds.qcfilter.set_test(var_name2, index=np.arange(300, 400, dtype=int), test_number=4)

    # Plot data
    display = TimeSeriesDisplay(ds, subplot_shape=(1, ), figsize=(10, 6))
    display.plot(var_name1, label=var_name1,
                 assessment_overplot=True, overplot_behind=True)
    display.plot(var_name2, day_night_background=True, color='green',
                 label=var_name2, assessment_overplot=True)

    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_barbs_from_u_v():
    sonde_ds = arm.read_netcdf(
        sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
    BarbDisplay = TimeSeriesDisplay({'sonde_darwin': sonde_ds})
    BarbDisplay.plot_barbs_from_u_v('u_wind', 'v_wind', 'pres',
                                    num_barbs_x=20)
    sonde_ds.close()
    return BarbDisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_barbs_from_u_v2():
    bins = list(np.linspace(0, 1, 10))
    xbins = list(pd.date_range(pd.to_datetime('2020-01-01'),
                               pd.to_datetime('2020-01-02'), 12))
    y_data = np.full([len(xbins), len(bins)], 1.)
    x_data = np.full([len(xbins), len(bins)], 2.)
    y_array = xr.DataArray(y_data, dims={'xbins': xbins, 'ybins': bins},
                           attrs={'units': 'm/s'})
    x_array = xr.DataArray(x_data, dims={'xbins': xbins, 'ybins': bins},
                           attrs={'units': 'm/s'})
    xbins = xr.DataArray(xbins, dims={'xbins': xbins})
    ybins = xr.DataArray(bins, dims={'ybins': bins})
    fake_obj = xr.Dataset({'xbins': xbins, 'ybins': ybins,
                          'ydata': y_array, 'xdata': x_array})
    BarbDisplay = TimeSeriesDisplay(fake_obj)
    BarbDisplay.plot_barbs_from_u_v('xdata', 'ydata', None, num_barbs_x=20,
                                    num_barbs_y=20, set_title='test plot', cmap='jet')
    fake_obj.close()
    return BarbDisplay.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_2D_timeseries_plot():
    obj = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)
    display = TimeSeriesDisplay(obj)
    display.plot('backscatter', y_rng=[0, 5000], use_var_for_y='range')
    matplotlib.pyplot.show()
    return display.fig
