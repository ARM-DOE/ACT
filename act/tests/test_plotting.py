import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime

import act
import act.io.armfiles as arm
import act.tests.sample_files as sample_files
from act.plotting import (
    ContourDisplay,
    GeographicPlotDisplay,
    SkewTDisplay,
    TimeSeriesDisplay,
    WindRoseDisplay,
    XSectionDisplay,
    DistributionDisplay,
)
from act.utils.data_utils import accumulate_precip

try:
    import cartopy
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

matplotlib.use('Agg')


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot():
    # Process MET data to get simple LCL
    files = sample_files.EXAMPLE_MET_WILDCARD
    met = arm.read_netcdf(files)
    met_temp = met.temp_mean
    met_rh = met.rh_mean
    met_lcl = (20.0 + met_temp / 5.0) * (100.0 - met_rh) / 1000.0
    met['met_lcl'] = met_lcl * 1000.0
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
    windrose.plot('wdir_vec_mean', 'wspd_vec_mean', spd_bins=np.linspace(0, 10, 4))
    windrose.axes[0].legend(loc='best')
    met.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


def test_errors():
    files = sample_files.EXAMPLE_MET_WILDCARD
    ds = arm.read_netcdf(files)

    display = TimeSeriesDisplay(ds)
    display.axes = None
    with np.testing.assert_raises(RuntimeError):
        display.day_night_background()

    display = TimeSeriesDisplay({'met': ds, 'met2': ds})
    with np.testing.assert_raises(ValueError):
        display.plot('temp_mean')
    with np.testing.assert_raises(ValueError):
        display.qc_flag_block_plot('qc_temp_mean')
    with np.testing.assert_raises(ValueError):
        display.plot_barbs_from_spd_dir('wdir_vec_mean', 'wspd_vec_mean')
    with np.testing.assert_raises(ValueError):
        display.plot_barbs_from_u_v('wdir_vec_mean', 'wspd_vec_mean')

    del ds.attrs['_file_dates']

    data = np.empty(len(ds['time'])) * np.nan
    lat = ds['lat'].values
    lon = ds['lon'].values
    ds['lat'].values = data
    ds['lon'].values = data

    display = TimeSeriesDisplay(ds)
    display.plot('temp_mean')
    display.set_yrng([0, 0])
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()
    ds['lat'].values = lat
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()
    ds['lon'].values = lon * 100.0
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()
    ds['lat'].values = lat * 100.0
    with np.testing.assert_warns(RuntimeWarning):
        display.day_night_background()

    ds.close()

    # Test some of the other errors
    ds = arm.read_netcdf(files)
    del ds['temp_mean'].attrs['units']
    display = TimeSeriesDisplay(ds)
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
    display = TimeSeriesDisplay(ds)
    display.add_subplots((2, 2), figsize=(15, 10))
    display.assign_to_figure_axis(fig, ax)
    assert display.fig is not None
    assert display.axes is not None

    ds = arm.read_netcdf(files)
    display = TimeSeriesDisplay(ds)
    ds.clean.cleanup()
    display.axes = None
    display.fig = None
    display.qc_flag_block_plot('atmos_pressure')
    assert display.fig is not None
    assert display.axes is not None

    matplotlib.pyplot.close(fig=display.fig)


def test_histogram_errors():
    files = sample_files.EXAMPLE_MET1
    ds = arm.read_netcdf(files)

    histdisplay = DistributionDisplay(ds)
    histdisplay.axes = None
    with np.testing.assert_raises(RuntimeError):
        histdisplay.set_yrng([0, 10])
    with np.testing.assert_raises(RuntimeError):
        histdisplay.set_xrng([-40, 40])
    histdisplay.fig = None
    histdisplay.plot_stacked_bar_graph('temp_mean', bins=np.arange(-40, 40, 5))
    histdisplay.set_yrng([0, 0])
    assert histdisplay.yrng[0][1] == 1.0
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    with np.testing.assert_raises(AttributeError):
        DistributionDisplay([])

    histdisplay.axes = None
    histdisplay.fig = None
    histdisplay.plot_stairstep_graph('temp_mean', bins=np.arange(-40, 40, 5))
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    sigma = 10
    mu = 50
    bins = np.linspace(0, 100, 50)
    ydata = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((bins - mu) ** 2) / (2 * sigma**2))
    y_array = xr.DataArray(ydata, dims={'time': bins})
    bins = xr.DataArray(bins, dims={'time': bins})
    my_fake_ds = xr.Dataset({'time': bins, 'ydata': y_array})
    histdisplay = DistributionDisplay(my_fake_ds)
    histdisplay.axes = None
    histdisplay.fig = None
    histdisplay.plot_size_distribution('ydata', 'time', set_title='Fake distribution.')
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)
    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.axes = None
    histdisplay.fig = None
    histdisplay.plot_heatmap(
        'tdry',
        'alt',
        x_bins=np.arange(-60, 10, 1),
        y_bins=np.linspace(0, 10000.0, 50),
        cmap='coolwarm',
    )
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    matplotlib.pyplot.close(fig=histdisplay.fig)


def test_xsection_errors():
    ds = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)

    display = XSectionDisplay(ds, figsize=(10, 8), subplot_shape=(2,))
    display.axes = None
    with np.testing.assert_raises(RuntimeError):
        display.set_yrng([0, 10])
    with np.testing.assert_raises(RuntimeError):
        display.set_xrng([-40, 40])

    display = XSectionDisplay(ds, figsize=(10, 8), subplot_shape=(1,))
    with np.testing.assert_raises(RuntimeError):
        display.plot_xsection(None, 'backscatter', x='time', cmap='HomeyerRainbow')

    ds.close()
    matplotlib.pyplot.close(fig=display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_tuple():
    ds = arm.read_netcdf(sample_files.EXAMPLE_MET1)
    ds2 = arm.read_netcdf(sample_files.EXAMPLE_SIRS)
    ds = ds.rename({'lat': 'fun_time'})
    ds['fun_time'].attrs['standard_name'] = 'latitude'
    ds = ds.rename({'lon': 'not_so_fun_time'})
    ds['not_so_fun_time'].attrs['standard_name'] = 'longitude'

    # You can use tuples if the datasets in the tuple contain a
    # datastream attribute. This is required in all ARM datasets.
    display = TimeSeriesDisplay((ds, ds2), subplot_shape=(2,), figsize=(15, 10))
    display.plot('short_direct_normal', 'sgpsirsE13.b1', subplot_index=(0,))
    display.day_night_background('sgpsirsE13.b1', subplot_index=(0,))
    display.plot('temp_mean', 'sgpmetE13.b1', subplot_index=(1,))
    display.day_night_background('sgpmetE13.b1', subplot_index=(1,))

    ax = act.plotting.common.parse_ax(ax=None)
    ax, fig = act.plotting.common.parse_ax_fig(ax=None, fig=None)
    ds.close()
    ds2.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_multidataset_plot_dict():

    ds = arm.read_netcdf(sample_files.EXAMPLE_MET1)
    ds2 = arm.read_netcdf(sample_files.EXAMPLE_SIRS)

    # You can use tuples if the datasets in the tuple contain a
    # datastream attribute. This is required in all ARM datasets.
    display = TimeSeriesDisplay({'sirs': ds2, 'met': ds}, subplot_shape=(2,), figsize=(15, 10))
    display.plot('short_direct_normal', 'sirs', subplot_index=(0,))
    display.day_night_background('sirs', subplot_index=(0,))
    display.plot('temp_mean', 'met', subplot_index=(1,))
    display.day_night_background('met', subplot_index=(1,))
    ds.close()
    ds2.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_wind_rose():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_TWP_SONDE_WILDCARD)

    WindDisplay = WindRoseDisplay(sonde_ds, figsize=(10, 10))
    WindDisplay.plot(
        'deg',
        'wspd',
        spd_bins=np.linspace(0, 20, 10),
        num_dirs=30,
        tick_interval=2,
        cmap='viridis',
    )
    WindDisplay.set_thetarng(trng=(0.0, 360.0))
    WindDisplay.set_rrng((0.0, 14))

    sonde_ds.close()

    try:
        return WindDisplay.fig
    finally:
        matplotlib.pyplot.close(WindDisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_barb_sounding_plot():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
    BarbDisplay = TimeSeriesDisplay({'sonde_darwin': sonde_ds})
    BarbDisplay.plot_time_height_xsection_from_1d_data(
        'rh', 'pres', cmap='coolwarm_r', vmin=0, vmax=100, num_time_periods=25
    )
    BarbDisplay.plot_barbs_from_spd_dir('wspd', 'deg', 'pres', num_barbs_x=20)
    sonde_ds.close()

    try:
        return BarbDisplay.fig
    finally:
        matplotlib.pyplot.close(BarbDisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_skewt_plot():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)
    skewt = SkewTDisplay(sonde_ds)
    skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
    sonde_ds.close()
    try:
        return skewt.fig
    finally:
        matplotlib.pyplot.close(skewt.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_skewt_plot_spd_dir():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)
    skewt = SkewTDisplay(sonde_ds, ds_name='act_datastream')
    skewt.plot_from_spd_and_dir('wspd', 'deg', 'pres', 'tdry', 'dp')
    sonde_ds.close()
    try:
        return skewt.fig
    finally:
        matplotlib.pyplot.close(skewt.fig)


@pytest.mark.mpl_image_compare(tolerance=81)
def test_multi_skewt_plot():

    files = glob.glob(sample_files.EXAMPLE_TWP_SONDE_20060121)
    test = {}
    for f in files:
        time = f.split('.')[-3]
        sonde_ds = arm.read_netcdf(f)
        sonde_ds = sonde_ds.resample(time='30s').nearest()
        test.update({time: sonde_ds})

    skewt = SkewTDisplay(test, subplot_shape=(2, 2))
    i = 0
    j = 0
    for f in files:
        time = f.split('.')[-3]
        skewt.plot_from_spd_and_dir(
            'wspd',
            'deg',
            'pres',
            'tdry',
            'dp',
            subplot_index=(j, i),
            dsname=time,
            p_levels_to_plot=np.arange(10.0, 1000.0, 25),
        )
        if j == 1:
            i += 1
            j = 0
        elif j == 0:
            j += 1
    try:
        return skewt.fig
    finally:
        matplotlib.pyplot.close(skewt.fig)


@pytest.mark.mpl_image_compare(tolerance=31)
def test_xsection_plot():
    visst_ds = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)

    xsection = XSectionDisplay(visst_ds, figsize=(10, 8))
    xsection.plot_xsection(
        None, 'backscatter', x='time', y='range', cmap='coolwarm', vmin=0, vmax=320
    )
    visst_ds.close()

    try:
        return xsection.fig
    finally:
        matplotlib.pyplot.close(xsection.fig)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed.")
@pytest.mark.mpl_image_compare(tolerance=30)
def test_xsection_plot_map():
    radar_ds = arm.read_netcdf(sample_files.EXAMPLE_VISST, combine='nested', concat_dim='time')

    try:
        xsection = XSectionDisplay(radar_ds, figsize=(15, 8))
        xsection.plot_xsection_map(
            None,
            'ir_temperature',
            vmin=220,
            vmax=300,
            cmap='Greys',
            x='longitude',
            y='latitude',
            isel_kwargs={'time': 0},
        )
        radar_ds.close()
        try:
            return xsection.fig
        finally:
            matplotlib.pyplot.close(xsection.fig)
    except Exception:
        pass


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason="Cartopy is not installed.")
@pytest.mark.mpl_image_compare(tolerance=30)
def test_geoplot():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)
    geodisplay = GeographicPlotDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    try:
        geodisplay.geoplot(
            'tdry',
            marker='.',
            cartopy_feature=[
                'STATES',
                'LAND',
                'OCEAN',
                'COASTLINE',
                'BORDERS',
                'LAKES',
                'RIVERS',
            ],
            text={'Ponca City': [-97.0725, 36.7125]},
        )
        try:
            return geodisplay.fig
        finally:
            matplotlib.pyplot.close(geodisplay.fig)
    except Exception:
        pass
    sonde_ds.close()


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stair_graph():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stairstep_graph('tdry', bins=np.arange(-60, 10, 1))
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stair_graph_sorted():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stairstep_graph(
        'tdry',
        bins=np.arange(-60, 10, 1),
        sortby_field='alt',
        sortby_bins=np.linspace(0, 10000.0, 6),
    )
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stacked_bar_graph():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar_graph('tdry', bins=np.arange(-60, 10, 1))
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stacked_bar_graph2():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar_graph('tdry')
    histdisplay.set_yrng([0, 400])
    histdisplay.set_xrng([-70, 0])
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_stacked_bar_graph_sorted():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar_graph(
        'tdry',
        bins=np.arange(-60, 10, 1),
        sortby_field='alt',
        sortby_bins=np.linspace(0, 10000.0, 6),
    )
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_heatmap():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_heatmap(
        'tdry',
        'alt',
        x_bins=np.arange(-60, 10, 1),
        y_bins=np.linspace(0, 10000.0, 50),
        cmap='coolwarm',
    )
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_size_distribution():
    sigma = 10
    mu = 50
    bins = np.linspace(0, 100, 50)
    ydata = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((bins - mu) ** 2) / (2 * sigma**2))
    y_array = xr.DataArray(ydata, dims={'time': bins})
    bins = xr.DataArray(bins, dims={'time': bins})
    my_fake_ds = xr.Dataset({'time': bins, 'ydata': y_array})
    histdisplay = DistributionDisplay(my_fake_ds)
    histdisplay.plot_size_distribution('ydata', 'time', set_title='Fake distribution.')
    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contour():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = arm.read_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contour', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=True, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='red')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contour_stamp():
    files = glob.glob(sample_files.EXAMPLE_STAMP_WILDCARD)
    test = {}
    stamp_fields = {}
    time = '2020-01-01T00:00:00.000000000'
    for f in files:
        ds = f.split('/')[-1]
        nc_ds = act.io.armfiles.read_netcdf(f)
        test.update({ds: nc_ds})
        stamp_fields.update({ds: ['lon', 'lat', 'plant_water_availability_east']})
        nc_ds.close()

    display = act.plotting.ContourDisplay(test, figsize=(8, 8))
    display.create_contour(fields=stamp_fields, time=time, levels=50, alpha=0.5, twod_dim_value=5)

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contour2():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = arm.read_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contour', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=False, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='pink')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contourf():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = arm.read_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update(
            {
                f: [
                    'lon',
                    'lat',
                    'atmos_pressure',
                    'temp_mean',
                    'rh_mean',
                    'vapor_pressure_mean',
                    'temp_std',
                ]
            }
        )

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contourf', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=True, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='red')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_contourf2():
    files = glob.glob(sample_files.EXAMPLE_MET_CONTOUR)
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = arm.read_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update(
            {
                f: [
                    'lon',
                    'lat',
                    'atmos_pressure',
                    'temp_mean',
                    'rh_mean',
                    'vapor_pressure_mean',
                    'temp_std',
                ]
            }
        )

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contourf', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=False, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='pink')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


# Due to issues with pytest-mpl, for now we just test to see if it runs
def test_time_height_scatter():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)

    display = TimeSeriesDisplay({'sgpsondewnpnC1.b1': sonde_ds}, figsize=(7, 3))
    display.time_height_scatter('tdry', day_night_background=False)

    sonde_ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_qc_bar_plot():
    ds = arm.read_netcdf(sample_files.EXAMPLE_MET1)
    ds.clean.cleanup()
    var_name = 'temp_mean'
    ds.qcfilter.set_test(var_name, index=range(100, 600), test_number=2)

    # Testing out when the assessment is not listed
    ds.qcfilter.set_test(var_name, index=range(500, 800), test_number=4)
    ds['qc_' + var_name].attrs['flag_assessments'][3] = 'Wonky'

    display = TimeSeriesDisplay({'sgpmetE13.b1': ds}, subplot_shape=(2,), figsize=(7, 4))
    display.plot(var_name, subplot_index=(0,), assessment_overplot=True)
    display.day_night_background('sgpmetE13.b1', subplot_index=(0,))
    color_lookup = {
        'Bad': 'red',
        'Incorrect': 'red',
        'Indeterminate': 'orange',
        'Suspect': 'orange',
        'Missing': 'darkgray',
        'Not Failing': 'green',
        'Acceptable': 'green',
    }
    display.qc_flag_block_plot(var_name, subplot_index=(1,), assessment_color=color_lookup)

    ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_2d_as_1d():
    ds = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)

    display = TimeSeriesDisplay(ds)
    display.plot('backscatter', force_line_plot=True, linestyle='None')

    ds.close()
    del ds

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_fill_between():
    ds = arm.read_netcdf(sample_files.EXAMPLE_MET_WILDCARD)

    accumulate_precip(ds, 'tbrg_precip_total')

    display = TimeSeriesDisplay(ds)
    display.fill_between('tbrg_precip_total_accumulated', color='gray', alpha=0.2)

    ds.close()
    del ds

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_qc_flag_block_plot():
    ds = arm.read_netcdf(sample_files.EXAMPLE_SURFSPECALB1MLAWER)

    display = TimeSeriesDisplay(ds, subplot_shape=(2,), figsize=(8, 2 * 4))

    display.plot('surface_albedo_mfr_narrowband_10m', force_line_plot=True, labels=True)

    display.qc_flag_block_plot('surface_albedo_mfr_narrowband_10m', subplot_index=(1,))

    ds.close()
    del ds

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


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
    display = TimeSeriesDisplay(ds, subplot_shape=(1,), figsize=(10, 6))
    display.plot(var_name, day_night_background=True, assessment_overplot=True)

    ds.close()
    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


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
    display = TimeSeriesDisplay(ds, subplot_shape=(1,), figsize=(10, 6))
    display.plot(
        var_name1, label=var_name1, assessment_overplot=True, overplot_behind=True, linestyle=''
    )
    display.plot(
        var_name2,
        day_night_background=True,
        color='green',
        label=var_name2,
        assessment_overplot=True,
        linestyle='',
    )

    ds.close()
    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_barbs_from_u_v():
    sonde_ds = arm.read_netcdf(sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
    BarbDisplay = TimeSeriesDisplay({'sonde_darwin': sonde_ds})
    BarbDisplay.plot_barbs_from_u_v('u_wind', 'v_wind', 'pres', num_barbs_x=20)
    sonde_ds.close()
    try:
        return BarbDisplay.fig
    finally:
        matplotlib.pyplot.close(BarbDisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_barbs_from_u_v2():
    bins = list(np.linspace(0, 1, 10))
    xbins = list(pd.date_range(pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-02'), 12))
    y_data = np.full([len(xbins), len(bins)], 1.0)
    x_data = np.full([len(xbins), len(bins)], 2.0)
    y_array = xr.DataArray(y_data, dims={'xbins': xbins, 'ybins': bins}, attrs={'units': 'm/s'})
    x_array = xr.DataArray(x_data, dims={'xbins': xbins, 'ybins': bins}, attrs={'units': 'm/s'})
    xbins = xr.DataArray(xbins, dims={'xbins': xbins})
    ybins = xr.DataArray(bins, dims={'ybins': bins})
    fake_ds = xr.Dataset({'xbins': xbins, 'ybins': ybins, 'ydata': y_array, 'xdata': x_array})
    BarbDisplay = TimeSeriesDisplay(fake_ds)
    BarbDisplay.plot_barbs_from_u_v(
        'xdata',
        'ydata',
        None,
        num_barbs_x=20,
        num_barbs_y=20,
        set_title='test plot',
        cmap='jet',
    )
    fake_ds.close()
    try:
        return BarbDisplay.fig
    finally:
        matplotlib.pyplot.close(BarbDisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_2D_timeseries_plot():
    ds = arm.read_netcdf(sample_files.EXAMPLE_CEIL1)
    display = TimeSeriesDisplay(ds)
    display.plot('backscatter', y_rng=[0, 5000], use_var_for_y='range')
    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_time_plot():
    files = sample_files.EXAMPLE_MET1
    ds = arm.read_netcdf(files)
    display = TimeSeriesDisplay(ds)
    display.plot('time')
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_time_plot_match_color_ylabel():
    files = sample_files.EXAMPLE_MET1
    ds = arm.read_netcdf(files)
    display = TimeSeriesDisplay(ds)
    display.plot('time', match_line_label_color=True)
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=40)
def test_time_plot2():
    files = sample_files.EXAMPLE_MET1
    ds = arm.read_netcdf(files, decode_times=False, use_cftime=False)
    display = TimeSeriesDisplay(ds)
    display.plot('time')
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_y_axis_flag_meanings():
    variable = 'detection_status'
    ds = arm.read_netcdf(
        sample_files.EXAMPLE_CEIL1, keep_variables=[variable, 'lat', 'lon', 'alt']
    )
    ds.clean.clean_arm_state_variables(variable, override_cf_flag=True)

    display = TimeSeriesDisplay(ds, figsize=(12, 8), subplot_shape=(1,))
    display.plot(variable, subplot_index=(0,), day_night_background=True, y_axis_flag_meanings=18)
    display.fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.94)

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=35)
def test_colorbar_labels():
    variable = 'cloud_phase_hsrl'
    ds = arm.read_netcdf(sample_files.EXAMPLE_CLOUDPHASE)
    ds.clean.clean_arm_state_variables(variable)

    display = TimeSeriesDisplay(ds, figsize=(12, 8), subplot_shape=(1,))

    y_axis_labels = {}
    flag_colors = ['white', 'green', 'blue', 'red', 'cyan', 'orange', 'yellow', 'black', 'gray']
    for value, meaning, color in zip(
        ds[variable].attrs['flag_values'], ds[variable].attrs['flag_meanings'], flag_colors
    ):
        y_axis_labels[value] = {'text': meaning, 'color': color}

    display.plot(variable, subplot_index=(0,), colorbar_labels=y_axis_labels, cbar_h_adjust=0)
    display.fig.subplots_adjust(left=0.08, right=0.88, bottom=0.1, top=0.94)

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_plot_datarose():
    files = glob.glob(sample_files.EXAMPLE_MET_WILDCARD)
    ds = arm.read_netcdf(files)
    display = act.plotting.WindRoseDisplay(ds, subplot_shape=(2, 3), figsize=(16, 10))
    display.plot_data(
        'wdir_vec_mean',
        'wspd_vec_mean',
        'temp_mean',
        num_dirs=12,
        plot_type='line',
        subplot_index=(0, 0),
    )
    display.plot_data(
        'wdir_vec_mean',
        'wspd_vec_mean',
        'temp_mean',
        num_dirs=12,
        plot_type='line',
        subplot_index=(0, 1),
        line_plot_calc='median',
    )
    display.plot_data(
        'wdir_vec_mean',
        'wspd_vec_mean',
        'temp_mean',
        num_dirs=12,
        plot_type='line',
        subplot_index=(0, 2),
        line_plot_calc='stdev',
    )
    display.plot_data(
        'wdir_vec_mean',
        'wspd_vec_mean',
        'temp_mean',
        num_dirs=12,
        plot_type='contour',
        subplot_index=(1, 0),
    )
    display.plot_data(
        'wdir_vec_mean',
        'wspd_vec_mean',
        'temp_mean',
        num_dirs=12,
        plot_type='contour',
        contour_type='mean',
        num_data_bins=10,
        clevels=21,
        cmap='rainbow',
        vmin=-5,
        vmax=20,
        subplot_index=(1, 1),
    )
    display.plot_data(
        'wdir_vec_mean',
        'wspd_vec_mean',
        'temp_mean',
        num_dirs=12,
        plot_type='boxplot',
        subplot_index=(1, 2),
    )

    display2 = act.plotting.WindRoseDisplay(
        {'ds1': ds, 'ds2': ds}, subplot_shape=(2, 3), figsize=(16, 10)
    )
    with np.testing.assert_raises(ValueError):
        display2.plot_data(
            'wdir_vec_mean',
            'wspd_vec_mean',
            'temp_mean',
            dsname='ds1',
            num_dirs=12,
            plot_type='line',
            line_plot_calc='T',
            subplot_index=(0, 0),
        )
    with np.testing.assert_raises(ValueError):
        display2.plot_data(
            'wdir_vec_mean',
            'wspd_vec_mean',
            'temp_mean',
            num_dirs=12,
            plot_type='line',
            subplot_index=(0, 0),
        )
    with np.testing.assert_raises(ValueError):
        display2.plot_data(
            'wdir_vec_mean',
            'wspd_vec_mean',
            'temp_mean',
            num_dirs=12,
            plot_type='groovy',
            subplot_index=(0, 0),
        )

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_add_nan_line():
    ds = arm.read_netcdf(sample_files.EXAMPLE_MET1)

    index = (ds.time.values <= np.datetime64('2019-01-01 04:00:00')) | (
        ds.time.values >= np.datetime64('2019-01-01 06:00:00')
    )
    ds = ds.sel({'time': index})

    index = (ds.time.values <= np.datetime64('2019-01-01 18:34:00')) | (
        ds.time.values >= np.datetime64('2019-01-01 19:06:00')
    )
    ds = ds.sel({'time': index})

    index = (ds.time.values <= np.datetime64('2019-01-01 12:30:00')) | (
        ds.time.values >= np.datetime64('2019-01-01 12:40:00')
    )
    ds = ds.sel({'time': index})

    display = TimeSeriesDisplay(ds, figsize=(15, 10), subplot_shape=(1,))
    display.plot('temp_mean', subplot_index=(0,), add_nan=True, day_night_background=True)
    ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_timeseries_invert():
    ds = arm.read_netcdf(sample_files.EXAMPLE_IRT25m20s)
    display = TimeSeriesDisplay(ds, figsize=(10, 8))
    display.plot('inst_sfc_ir_temp', invert_y_axis=True)
    ds.close()
    return display.fig


def test_plot_time_rng():
    # Test if setting the xrange can be done with pandas or datetime datatype
    # eventhough the data is numpy. Check for correctly converting xrange values
    # before setting and not causing an exception.
    met = arm.read_netcdf(sample_files.EXAMPLE_MET1)

    # Plot data
    xrng = [datetime(2019, 1, 1, 0, 0), datetime(2019, 1, 2, 0, 0)]
    display = TimeSeriesDisplay(met)
    display.plot('temp_mean', time_rng=xrng)

    xrng = [pd.to_datetime('2019-01-01'), pd.to_datetime('2019-01-02')]
    display = TimeSeriesDisplay(met)
    display.plot('temp_mean', time_rng=xrng)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_groupby_plot():
    ds = arm.read_netcdf(act.tests.EXAMPLE_MET_WILDCARD)

    # Create Plot Display
    display = WindRoseDisplay(ds, figsize=(15, 15), subplot_shape=(3, 3))
    groupby = display.group_by('day')
    groupby.plot_group('plot_data', None, dir_field='wdir_vec_mean', spd_field='wspd_vec_mean',
                       data_field='temp_mean', num_dirs=12, plot_type='line')

    # Set theta tick markers for each axis inside display to be inside the polar axes
    for i in range(3):
        for j in range(3):
            display.axes[i, j].tick_params(pad=-20)
    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_match_ylimits_plot():
    files = glob.glob(sample_files.EXAMPLE_MET_WILDCARD)
    ds = arm.read_netcdf(files)
    display = act.plotting.TimeSeriesDisplay(ds, figsize=(10, 8),
                                             subplot_shape=(2, 2))
    groupby = display.group_by('day')
    groupby.plot_group('plot', None, field='temp_mean', marker=' ')
    groupby.display.set_yrng([0, 20], match_axes_ylimits=True)
    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_enhanced_skewt_plot():
    ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)
    display = act.plotting.SkewTDisplay(ds)
    display.plot_enhanced_skewt(color_field='alt', component_range=85)
    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_enhanced_skewt_plot_2():
    ds = arm.read_netcdf(sample_files.EXAMPLE_SONDE1)
    display = act.plotting.SkewTDisplay(ds)
    overwrite_data = {'Test': 1234.}
    display.plot_enhanced_skewt(spd_name='u_wind', dir_name='v_wind',
                                color_field='alt', component_range=85, uv_flag=True,
                                overwrite_data=overwrite_data, add_data=overwrite_data)
    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_xlim_correction_plot():
    ds = arm.read_netcdf(sample_files.EXAMPLE_MET1)

    # Plot data
    xrng = [datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0)]
    display = TimeSeriesDisplay(ds)
    display.plot('temp_mean', time_rng=xrng)

    ds.close()

    return display.fig


def test_histogram_kwargs():
    files = sample_files.EXAMPLE_MET1
    ds = arm.read_netcdf(files)
    hist_kwargs = {'range': (-10, 10)}
    histdisplay = DistributionDisplay(ds)
    hist_dict = histdisplay.plot_stacked_bar_graph('temp_mean', bins=np.arange(-40, 40, 5),
                                                   sortby_bins=np.arange(-40, 40, 5),
                                                   hist_kwargs=hist_kwargs)
    hist_array = np.array(
        [0, 0, 0, 0, 0, 0, 493, 883, 64, 0, 0, 0, 0, 0, 0])
    assert_allclose(hist_dict['histogram'], hist_array)
    hist_dict = histdisplay.plot_stacked_bar_graph('temp_mean', hist_kwargs=hist_kwargs)
    hist_array = np.array([0, 0, 950, 177, 249, 64, 0, 0, 0, 0])
    assert_allclose(hist_dict['histogram'], hist_array)

    hist_dict_stair = histdisplay.plot_stairstep_graph('temp_mean', bins=np.arange(-40, 40, 5),
                                                       sortby_bins=np.arange(-40, 40, 5),
                                                       hist_kwargs=hist_kwargs)
    hist_array = np.array(
        [0, 0, 0, 0, 0, 0, 493, 883, 64, 0, 0, 0, 0, 0, 0])
    assert_allclose(hist_dict_stair['histogram'], hist_array)
    hist_dict_stair = histdisplay.plot_stairstep_graph('temp_mean', hist_kwargs=hist_kwargs)
    hist_array = np.array([0, 0, 950, 177, 249, 64, 0, 0, 0, 0])
    assert_allclose(hist_dict_stair['histogram'], hist_array)

    hist_dict_heat = histdisplay.plot_heatmap('temp_mean', 'rh_mean', x_bins=np.arange(-60, 10, 1),
                                              y_bins=np.linspace(0, 10000.0, 50),
                                              hist_kwargs=hist_kwargs)
    hist_array = [0.0, 0.0, 0.0, 0.0]
    assert_allclose(hist_dict_heat['histogram'][0, 0:4], hist_array)
    ds.close()
    matplotlib.pyplot.close(fig=histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=30)
def test_violin():
    ds = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_MET1)

    # Create a DistributionDisplay object to compare fields
    display = DistributionDisplay(ds)

    # Create violin display of mean temperature
    display.plot_violin('temp_mean',
                        positions=[5.0],
                        set_title='SGP MET E13 2019-01-01'
                        )

    ds.close()

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=30)
def test_scatter():
    ds = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_MET1)
    # Create a DistributionDisplay object to compare fields
    display = DistributionDisplay(ds)

    display.plot_scatter('wspd_arith_mean',
                         'wspd_vec_mean',
                         m_field='wdir_vec_mean',
                         marker='d',
                         cmap='bwr')
    # Set the range of the field on the x-axis
    display.set_xrng((0, 14))
    display.set_yrng((0, 14))
    # Display the 1:1 ratio line
    display.set_ratio_line()

    ds.close()

    return display.fig
