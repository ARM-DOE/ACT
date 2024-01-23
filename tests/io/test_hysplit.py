import act
import matplotlib.pyplot as plt

from arm_test_data import DATASETS


def test_read_hysplit():
    filename = DATASETS.fetch('houstonaug300.0summer2010080100')
    ds = act.io.read_hysplit(filename)
    assert 'lat' in ds.variables.keys()
    assert 'lon' in ds.variables.keys()
    assert 'alt' in ds.variables.keys()
    assert 'PRESSURE' in ds.variables.keys()
    assert ds.dims["num_grids"] == 8
    assert ds.dims["num_trajectories"] == 1
    assert ds.dims['time'] == 121
    assert ds['age'].min() == -120

