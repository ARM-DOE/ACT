import numpy as np
import act


def test_get_improve():
    ds = act.discovery.get_improve_data(site_id='244', start_date='1/1/2023', end_date='12/31/2023')

    assert len(list(ds)) == 216
    assert 'lat' in ds
    assert 'lon' in ds
    assert len(ds.time.values) == 121
    assert 'aluminum_fine' in ds
    assert ds['ammonium_nitrate_fine'].values[0] == 1.41363

    with np.testing.assert_raises(ValueError):
        ds = act.discovery.get_improve_data()
    with np.testing.assert_raises(ValueError):
        ds = act.discovery.get_improve_data(site_id='244')
    with np.testing.assert_raises(ValueError):
        ds = act.discovery.get_improve_data(site_id='244', start_date='1/1/2023')
