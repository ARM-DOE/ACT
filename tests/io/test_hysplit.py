import act

from act.tests import sample_files


def test_read_hysplit():
    filename = sample_files.EXAMPLE_HYSPLIT
    ds = act.io.read_hysplit(filename)
    assert 'lat' in ds.variables.keys()
    assert 'lon' in ds.variables.keys()
    assert 'alt' in ds.variables.keys()
    assert 'PRESSURE' in ds.variables.keys()
    assert ds.sizes["num_grids"] == 8
    assert ds.sizes["num_trajectories"] == 1
    assert ds.sizes['time'] == 120
    assert ds['age'].min() == -120
