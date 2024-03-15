import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import act
from act.plotting import SkewTDisplay
from act.tests import sample_files

matplotlib.use('Agg')


@pytest.mark.mpl_image_compare(tolerance=10)
def test_skewt_plot():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    skewt = SkewTDisplay(sonde_ds)
    skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
    sonde_ds.close()
    return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_skewt_plot_spd_dir():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    skewt = SkewTDisplay(sonde_ds, ds_name='act_datastream')
    skewt.plot_from_spd_and_dir('wspd', 'deg', 'pres', 'tdry', 'dp')
    sonde_ds.close()
    return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_multi_skewt_plot():
    files = sample_files.EXAMPLE_TWP_SONDE_20060121
    test = {}
    for f in files:
        time = f.split('.')[-3]
        sonde_ds = act.io.arm.read_arm_netcdf(f)
        sonde_ds = sonde_ds.resample(time='30s').nearest()
        test.update({time: sonde_ds})

    skewt = SkewTDisplay(test, subplot_shape=(2, 2), figsize=(12, 14))
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
        skewt.axes[j, i].set_ylim([1000, 10])
        if j == 1:
            i += 1
            j = 0
        elif j == 0:
            j += 1
    plt.tight_layout()
    return skewt.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_enhanced_skewt_plot():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    display = act.plotting.SkewTDisplay(ds)
    display.plot_enhanced_skewt(color_field='alt', component_range=85)
    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_enhanced_skewt_plot_2():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    display = act.plotting.SkewTDisplay(ds)
    overwrite_data = {'Test': 1234.0}
    display.plot_enhanced_skewt(
        spd_name='u_wind',
        dir_name='v_wind',
        color_field='alt',
        component_range=85,
        uv_flag=True,
        overwrite_data=overwrite_data,
        add_data=overwrite_data,
    )
    ds.close()
    return display.fig


@pytest.mark.mpl_image_compare(tolerance=10)
def test_skewt_options():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    skewt = SkewTDisplay(sonde_ds)
    skewt.plot_from_u_and_v(
        'u_wind',
        'v_wind',
        'pres',
        'tdry',
        'dp',
        plot_dry_adiabats=True,
        plot_moist_adiabats=True,
        plot_mixing_lines=True,
    )
    sonde_ds.close()
    return skewt.fig
