import matplotlib

import pytest

import act
from act.tests import sample_files
from act.plotting import GeographicPlotDisplay

try:
    import cartopy

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

matplotlib.use('Agg')


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason='Cartopy is not installed.')
@pytest.mark.mpl_image_compare(style="default", tolerance=30)
def test_geoplot():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    geodisplay = GeographicPlotDisplay({'sgpsondewnpnC1.b1': sonde_ds}, figsize=(15, 8))
    try:
        geodisplay.geoplot(
            'tdry',
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
            img_tile=None,
        )
        try:
            return geodisplay.fig
        finally:
            matplotlib.pyplot.close(geodisplay.fig)
    except Exception:
        pass
    sonde_ds.close()


@pytest.mark.skipif(not CARTOPY_AVAILABLE, reason='Cartopy is not installed.')
@pytest.mark.mpl_image_compare(style="default", tolerance=30)
def test_geoplot_tile():
    sonde_ds = act.io.arm.read_arm_netcdf(sample_files.EXAMPLE_SONDE1)
    geodisplay = GeographicPlotDisplay({'sgpsondewnpnC1.b1': sonde_ds}, figsize=(15, 8))
    try:
        geodisplay.geoplot(
            'tdry',
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
            img_tile='GoogleTiles',
            img_tile_args={'style': 'street'},
        )
        try:
            return geodisplay.fig
        finally:
            matplotlib.pyplot.close(geodisplay.fig)
    except Exception:
        pass
    sonde_ds.close()
