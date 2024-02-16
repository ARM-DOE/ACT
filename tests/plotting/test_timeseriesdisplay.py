from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import act
from act.plotting import TimeSeriesDisplay, WindRoseDisplay
from act.tests import sample_files
from act.utils.data_utils import accumulate_precip

matplotlib.use('Agg')


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot():
    # Process MET data to get simple LCL
    files = sample_files.EXAMPLE_MET_WILDCARD
    met = act.io.arm.read_arm_netcdf(files)
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
    ds = act.io.arm.read_arm_netcdf(files)

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
    with np.testing.assert_raises(ValueError):
        display.plot_time_height_xsection_from_1d_data('wdir_vec_mean', 'wspd_vec_mean')
    with np.testing.assert_raises(ValueError):
        display.time_height_scatter('wdir_vec_mean')

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
    ds = act.io.arm.read_arm_netcdf(files)
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

    ds = act.io.arm.read_arm_netcdf(files)
    display = TimeSeriesDisplay(ds)
    ds.clean.cleanup()
    display.axes = None
    display.fig = None
    display.qc_flag_block_plot('atmos_pressure')
    assert display.fig is not None
    assert display.axes is not None

    matplotlib.pyplot.close(fig=display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_multidataset_plot_tuple():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)
    ds2 = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SIRS)
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_multidataset_plot_dict():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)
    ds2 = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SIRS)

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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_barb_sounding_plot():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
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


# Due to issues with pytest-mpl, for now we just test to see if it runs
@pytest.mark.mpl_image_compare(tolerance=10)
def test_time_height_scatter():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

    display = TimeSeriesDisplay({'sgpsondewnpnC1.b1': sonde_ds}, figsize=(10, 6))
    display.time_height_scatter('tdry', plot_alt_field=True)

    sonde_ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


# Due to issues with pytest-mpl, for now we just test to see if it runs
@pytest.mark.mpl_image_compare(tolerance=10)
def test_time_height_scatter2():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

    display = TimeSeriesDisplay(
        {'sgpsondewnpnC1.b1': sonde_ds}, figsize=(8, 10), subplot_shape=(2,)
    )
    display.time_height_scatter(
        'tdry', day_night_background=True, subplot_index=(0,), cb_friendly=True, plot_alt_field=True
    )
    display.time_height_scatter(
        'rh', day_night_background=True, subplot_index=(1,), cb_friendly=True
    )

    sonde_ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_qc_bar_plot():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_2d_as_1d():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_CEIL1)

    display = TimeSeriesDisplay(ds)
    display.plot('backscatter', force_line_plot=True, linestyle='None')

    ds.close()
    del ds

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_fill_between():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET_WILDCARD)

    accumulate_precip(ds, 'tbrg_precip_total')

    display = TimeSeriesDisplay(ds)
    display.fill_between('tbrg_precip_total_accumulated', color='gray', alpha=0.2)

    ds.close()
    del ds

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_qc_flag_block_plot():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SURFSPECALB1MLAWER)

    display = TimeSeriesDisplay(ds, subplot_shape=(2,), figsize=(10, 8))

    display.plot('surface_albedo_mfr_narrowband_10m', force_line_plot=True, labels=True)

    display.qc_flag_block_plot(
        'surface_albedo_mfr_narrowband_10m', subplot_index=(1,), cb_friendly=True
    )

    ds.close()
    del ds

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_assessment_overplot():
    var_name = 'temp_mean'
    files = sample_files.EXAMPLE_MET1
    ds = act.io.arm.read_arm_netcdf(files)
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_assessment_overplot_multi():
    var_name1, var_name2 = 'wspd_arith_mean', 'wspd_vec_mean'
    files = sample_files.EXAMPLE_MET1
    ds = act.io.arm.read_arm_netcdf(files)
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_barbs_from_u_v():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
    BarbDisplay = TimeSeriesDisplay({'sonde_darwin': sonde_ds})
    BarbDisplay.plot_barbs_from_u_v('u_wind', 'v_wind', 'pres', num_barbs_x=20)
    sonde_ds.close()
    try:
        return BarbDisplay.fig
    finally:
        matplotlib.pyplot.close(BarbDisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
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


def test_plot_barbs_from_u_v3():
    bins = list(np.linspace(0, 1, 10))
    xbins = list(pd.date_range(pd.to_datetime('2020-01-01'), pd.to_datetime('2020-01-02'), 12))
    y_data = np.full([len(xbins), len(bins)], 1.0)
    x_data = np.full([len(xbins), len(bins)], 2.0)
    pres = np.linspace(1000, 0, len(bins))
    y_array = xr.DataArray(y_data, dims={'xbins': xbins, 'ybins': bins}, attrs={'units': 'm/s'})
    x_array = xr.DataArray(x_data, dims={'xbins': xbins, 'ybins': bins}, attrs={'units': 'm/s'})
    xbins = xr.DataArray(xbins, dims={'xbins': xbins})
    ybins = xr.DataArray(bins, dims={'ybins': bins})
    pres = xr.DataArray(pres, dims={'ybins': bins}, attrs={'units': 'hPa'})
    fake_ds = xr.Dataset(
        {'xbins': xbins, 'ybins': ybins, 'ydata': y_array, 'xdata': x_array, 'pres': pres}
    )
    BarbDisplay = TimeSeriesDisplay(fake_ds)
    BarbDisplay.plot_barbs_from_u_v('xdata', 'ydata', None, set_title='test', use_var_for_y='pres')
    fake_ds.close()
    try:
        return BarbDisplay.fig
    finally:
        matplotlib.pyplot.close(BarbDisplay.fig)


def test_plot_barbs_from_u_v4():
    bins = list(np.linspace(0, 1, 10))
    xbins = [pd.to_datetime('2020-01-01')]
    y_data = np.full([1], 1.0)
    x_data = np.full([1], 2.0)
    pres = np.linspace(1000, 0, len(bins))
    y_array = xr.DataArray(y_data, dims={'xbins': xbins}, attrs={'units': 'm/s'})
    x_array = xr.DataArray(x_data, dims={'xbins': xbins}, attrs={'units': 'm/s'})
    xbins = xr.DataArray(xbins, dims={'xbins': xbins})
    ybins = xr.DataArray(bins, dims={'ybins': bins})
    pres = xr.DataArray(pres, dims={'ybins': bins}, attrs={'units': 'hPa'})
    fake_ds = xr.Dataset(
        {'xbins': xbins, 'ybins': ybins, 'ydata': y_array, 'xdata': x_array, 'pres': pres}
    )
    BarbDisplay = TimeSeriesDisplay(fake_ds)
    BarbDisplay.plot_barbs_from_u_v(
        'xdata', 'ydata', None, set_title='test', use_var_for_y='pres', cmap='jet'
    )
    fake_ds.close()
    try:
        return BarbDisplay.fig
    finally:
        matplotlib.pyplot.close(BarbDisplay.fig)


def test_plot_barbs_from_u_v5():
    bins = list(np.linspace(0, 1, 10))
    xbins = [pd.to_datetime('2020-01-01')]
    y_data = np.full([1], 1.0)
    x_data = np.full([1], 2.0)
    pres = np.linspace(1000, 0, len(bins))
    y_array = xr.DataArray(y_data, dims={'xbins': xbins}, attrs={'units': 'm/s'})
    x_array = xr.DataArray(x_data, dims={'xbins': xbins}, attrs={'units': 'm/s'})
    xbins = xr.DataArray(xbins, dims={'xbins': xbins})
    ybins = xr.DataArray(bins, dims={'ybins': bins})
    pres = xr.DataArray(pres, dims={'ybins': bins}, attrs={'units': 'hPa'})
    fake_ds = xr.Dataset(
        {'xbins': xbins, 'ybins': ybins, 'ydata': y_array, 'xdata': x_array, 'pres': pres}
    )
    BarbDisplay = TimeSeriesDisplay(fake_ds)
    BarbDisplay.plot_barbs_from_u_v(
        'xdata',
        'ydata',
        None,
        set_title='test',
        use_var_for_y='pres',
    )
    fake_ds.close()
    try:
        return BarbDisplay.fig
    finally:
        matplotlib.pyplot.close(BarbDisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_2D_timeseries_plot():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_CEIL1)
    display = TimeSeriesDisplay(ds)
    display.plot('backscatter', y_rng=[0, 5000], use_var_for_y='range')
    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_time_plot():
    files = sample_files.EXAMPLE_MET1
    ds = act.io.arm.read_arm_netcdf(files)
    display = TimeSeriesDisplay(ds)
    display.plot('time')
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_time_plot_match_color_ylabel():
    files = sample_files.EXAMPLE_MET1
    ds = act.io.arm.read_arm_netcdf(files)
    display = TimeSeriesDisplay(ds)
    display.plot('time', match_line_label_color=True)
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_time_plot2():
    files = sample_files.EXAMPLE_MET1
    ds = act.io.arm.read_arm_netcdf(files, decode_times=False, use_cftime=False)
    display = TimeSeriesDisplay(ds)
    display.plot('time')
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_y_axis_flag_meanings():
    variable = 'detection_status'
    ds = act.io.arm.read_arm_netcdf(
        sample_files.EXAMPLE_CEIL1, keep_variables=[variable, 'lat', 'lon', 'alt']
    )
    ds.clean.clean_arm_state_variables(variable, override_cf_flag=True)

    display = TimeSeriesDisplay(ds, figsize=(12, 8), subplot_shape=(1,))
    display.plot(variable, subplot_index=(0,), day_night_background=True, y_axis_flag_meanings=18)
    display.fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.94)

    return display.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_colorbar_labels():
    variable = 'cloud_phase_hsrl'
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_CLOUDPHASE)
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_add_nan_line():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)

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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_timeseries_invert():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_IRT25m20s)
    display = TimeSeriesDisplay(ds, figsize=(10, 8))
    display.plot('inst_sfc_ir_temp', invert_y_axis=True)
    ds.close()
    return display.fig


def test_plot_time_rng():
    # Test if setting the xrange can be done with pandas or datetime datatype
    # eventhough the data is numpy. Check for correctly converting xrange values
    # before setting and not causing an exception.
    met = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)

    # Plot data
    xrng = [datetime(2019, 1, 1, 0, 0), datetime(2019, 1, 2, 0, 0)]
    display = TimeSeriesDisplay(met)
    display.plot('temp_mean', time_rng=xrng)

    xrng = [pd.to_datetime('2019-01-01'), pd.to_datetime('2019-01-02')]
    display = TimeSeriesDisplay(met)
    display.plot('temp_mean', time_rng=xrng)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_match_ylimits_plot():
    files = sample_files.EXAMPLE_MET_WILDCARD
    ds = act.io.arm.read_arm_netcdf(files)
    display = act.plotting.TimeSeriesDisplay(ds, figsize=(14, 8), subplot_shape=(2, 2))
    groupby = display.group_by('day')
    groupby.plot_group('plot', None, field='temp_mean', marker=' ')
    groupby.display.set_yrng([-20, 20], match_axes_ylimits=True)
    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_xlim_correction_plot():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)

    # Plot data
    xrng = [datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0)]
    display = TimeSeriesDisplay(ds)
    display.plot('temp_mean', time_rng=xrng)

    ds.close()

    return display.fig
