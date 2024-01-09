import numpy as np

import act


def test_sst():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_IRTSST)
    ds = act.retrievals.irt.sst_from_irt(ds)
    np.testing.assert_almost_equal(ds['sea_surface_temperature'].values[0], 278.901, decimal=3)
    np.testing.assert_almost_equal(ds['sea_surface_temperature'].values[-1], 279.291, decimal=3)
    assert np.round(np.nansum(ds['sea_surface_temperature'].values)).astype(int) == 6699
    ds.close()
