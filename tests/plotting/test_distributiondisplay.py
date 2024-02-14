import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import act
from act.plotting import DistributionDisplay
from act.tests import sample_files

matplotlib.use('Agg')


def test_distribution_errors():
    files = sample_files.EXAMPLE_MET1
    ds = act.io.arm.read_arm_netcdf(files)

    histdisplay = DistributionDisplay(ds)
    histdisplay.axes = None
    with np.testing.assert_raises(RuntimeError):
        histdisplay.set_yrng([0, 10])
    with np.testing.assert_raises(RuntimeError):
        histdisplay.set_xrng([-40, 40])
    histdisplay.fig = None
    histdisplay.plot_stacked_bar('temp_mean', bins=np.arange(-40, 40, 5))
    histdisplay.set_yrng([0, 0])
    assert histdisplay.yrng[0][1] == 1.0
    assert histdisplay.fig is not None
    assert histdisplay.axes is not None

    with np.testing.assert_raises(AttributeError):
        DistributionDisplay([])

    histdisplay.axes = None
    histdisplay.fig = None
    histdisplay.plot_stairstep('temp_mean', bins=np.arange(-40, 40, 5))
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

    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
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

    histdisplay = DistributionDisplay({'thing1': sonde_ds, 'thing2': sonde_ds})
    with np.testing.assert_raises(ValueError):
        histdisplay.plot_stacked_bar('tdry')
    with np.testing.assert_raises(ValueError):
        histdisplay.plot_size_distribution('tdry', 'time')
    with np.testing.assert_raises(ValueError):
        histdisplay.plot_stairstep('tdry')
    with np.testing.assert_raises(ValueError):
        histdisplay.plot_heatmap('tdry', 'alt')
    with np.testing.assert_raises(ValueError):
        histdisplay.plot_violin('tdry')
    matplotlib.pyplot.close(fig=histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_stair_graph():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stairstep('tdry', bins=np.arange(-60, 10, 1))
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_stair_graph2():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    del sonde_ds['tdry'].attrs['units']

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stairstep('tdry', sortby_field='alt')
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_stair_graph_sorted():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stairstep(
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_stacked_bar_graph():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar('tdry', bins=np.arange(-60, 10, 1))
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_stacked_bar_graph2():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar('tdry')
    histdisplay.set_yrng([0, 400])
    histdisplay.set_xrng([-70, 0])
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_stacked_bar_graph3():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    del sonde_ds['tdry'].attrs['units']

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar('tdry', sortby_field='alt')
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_stacked_bar_graph_sorted():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_stacked_bar(
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_heatmap():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)

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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_heatmap2():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    del sonde_ds['tdry'].attrs['units']

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_heatmap(
        'tdry',
        'alt',
        x_bins=10,
        y_bins=10,
        cmap='coolwarm',
    )
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_heatmap3():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    del sonde_ds['tdry'].attrs['units']

    histdisplay = DistributionDisplay({'sgpsondewnpnC1.b1': sonde_ds})
    histdisplay.plot_heatmap(
        'tdry',
        'alt',
        threshold=1,
        cmap='coolwarm',
    )
    sonde_ds.close()

    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_size_distribution2():
    sigma = 10
    mu = 50
    bins = pd.date_range('2023-01-01', '2023-01-02', periods=mu)
    ydata = (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((np.array(range(len(bins))) - mu) ** 2) / (2 * sigma**2))
    )
    y_array = xr.DataArray(ydata, dims={'time': bins})
    bins = xr.DataArray(bins, dims={'time': bins})
    my_fake_ds = xr.Dataset({'time': bins, 'ydata': y_array})
    my_fake_ds['ydata'].attrs['units'] = 'units'
    histdisplay = DistributionDisplay(my_fake_ds)
    histdisplay.plot_size_distribution('ydata', bins, time=bins.values[10])
    try:
        return histdisplay.fig
    finally:
        matplotlib.pyplot.close(histdisplay.fig)


def test_histogram_kwargs():
    files = sample_files.EXAMPLE_MET1
    ds = act.io.arm.read_arm_netcdf(files)
    hist_kwargs = {'range': (-10, 10)}
    histdisplay = DistributionDisplay(ds)
    hist_dict = histdisplay.plot_stacked_bar(
        'temp_mean',
        bins=np.arange(-40, 40, 5),
        sortby_bins=np.arange(-40, 40, 5),
        hist_kwargs=hist_kwargs,
    )
    hist_array = np.array([0, 0, 0, 0, 0, 0, 493, 883, 64, 0, 0, 0, 0, 0, 0])
    assert_allclose(hist_dict['histogram'], hist_array)
    hist_dict = histdisplay.plot_stacked_bar('temp_mean', hist_kwargs=hist_kwargs)
    hist_array = np.array([0, 0, 950, 177, 249, 64, 0, 0, 0, 0])
    assert_allclose(hist_dict['histogram'], hist_array)

    hist_dict_stair = histdisplay.plot_stairstep(
        'temp_mean',
        bins=np.arange(-40, 40, 5),
        sortby_bins=np.arange(-40, 40, 5),
        hist_kwargs=hist_kwargs,
    )
    hist_array = np.array([0, 0, 0, 0, 0, 0, 493, 883, 64, 0, 0, 0, 0, 0, 0])
    assert_allclose(hist_dict_stair['histogram'], hist_array)
    hist_dict_stair = histdisplay.plot_stairstep('temp_mean', hist_kwargs=hist_kwargs)
    hist_array = np.array([0, 0, 950, 177, 249, 64, 0, 0, 0, 0])
    assert_allclose(hist_dict_stair['histogram'], hist_array)

    hist_dict_heat = histdisplay.plot_heatmap(
        'temp_mean',
        'rh_mean',
        x_bins=np.arange(-60, 10, 1),
        y_bins=np.linspace(0, 10000.0, 50),
        hist_kwargs=hist_kwargs,
    )
    hist_array = [0.0, 0.0, 0.0, 0.0]
    assert_allclose(hist_dict_heat['histogram'][0, 0:4], hist_array)
    ds.close()
    matplotlib.pyplot.close(fig=histdisplay.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_violin():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)

    # Create a DistributionDisplay object to compare fields
    display = DistributionDisplay(ds)

    # Create violin display of mean temperature
    display.plot_violin('temp_mean', positions=[5.0], set_title='SGP MET E13 2019-01-01')

    ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_violin2():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)
    del ds['temp_mean'].attrs['units']

    # Create a DistributionDisplay object to compare fields
    display = DistributionDisplay(ds)

    # Create violin display of mean temperature
    display.plot_violin('temp_mean', vert=False)

    ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_scatter():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)
    # Create a DistributionDisplay object to compare fields
    display = DistributionDisplay(ds)

    display.plot_scatter(
        'wspd_arith_mean', 'wspd_vec_mean', m_field='wdir_vec_mean', marker='d', cmap='bwr'
    )
    # Set the range of the field on the x-axis
    display.set_xrng((0, 14))
    display.set_yrng((0, 14))
    # Display the 1:1 ratio line
    display.set_ratio_line()

    ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_scatter2():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_MET1)
    del ds['wspd_arith_mean'].attrs['units']
    del ds['wspd_vec_mean'].attrs['units']
    # Create a DistributionDisplay object to compare fields
    display = DistributionDisplay(ds)
    display.plot_scatter(
        'wspd_arith_mean',
        'wspd_vec_mean',
    )
    display.set_ratio_line()
    ds.close()

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)
