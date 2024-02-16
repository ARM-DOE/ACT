import matplotlib
import numpy as np
import pytest

import act
from act.plotting import WindRoseDisplay
from act.tests import sample_files

matplotlib.use('Agg')


@pytest.mark.mpl_image_compare(tolerance=10)
def test_wind_rose():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_TWP_SONDE_WILDCARD)

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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_datarose():
    files = sample_files.EXAMPLE_MET_WILDCARD
    ds = act.io.arm.read_arm_netcdf(files)
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


@pytest.mark.mpl_image_compare(tolerance=10)
def test_groupby_plot():
    ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MET_WILDCARD)

    # Create Plot Display
    display = WindRoseDisplay(ds, figsize=(15, 15), subplot_shape=(3, 3))
    groupby = display.group_by('day')
    groupby.plot_group(
        'plot_data',
        None,
        dir_field='wdir_vec_mean',
        spd_field='wspd_vec_mean',
        data_field='temp_mean',
        num_dirs=12,
        plot_type='line',
    )

    # Set theta tick markers for each axis inside display to be inside the polar axes
    for i in range(3):
        for j in range(3):
            display.axes[i, j].tick_params(pad=-20)
    ds.close()
    return display.fig
