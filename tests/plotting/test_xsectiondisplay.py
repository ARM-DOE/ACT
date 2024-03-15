import matplotlib
import numpy as np
import pytest

import act
from act.plotting import XSectionDisplay
from act.tests import sample_files

try:
    import cartopy  # noqa

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

matplotlib.use('Agg')


def test_xsection_errors():
    ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_CEIL1)

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


@pytest.mark.mpl_image_compare(tolerance=20)
def test_xsection_plot():
    visst_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_CEIL1)

    xsection = XSectionDisplay(visst_ds, figsize=(10, 8))
    xsection.plot_xsection(
        None, 'backscatter', x='time', y='range', cmap='coolwarm', vmin=0, vmax=320
    )
    visst_ds.close()

    try:
        return xsection.fig
    finally:
        matplotlib.pyplot.close(xsection.fig)


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason='Cartopy is not installed.')
@pytest.mark.mpl_image_compare(tolerance=10)
def test_xsection_plot_map():
    radar_ds = act.io.arm.read_arm_netcdf(
        sample_files.EXAMPLE_VISST, combine='nested', concat_dim='time'
    )
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
