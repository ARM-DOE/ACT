import numpy as np

import act


def test_aeri2irt():
    aeri_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_AERI)
    aeri_ds = act.retrievals.aeri.aeri2irt(aeri_ds)
    assert np.round(np.nansum(aeri_ds['aeri_irt_equiv_temperature'].values)).astype(int) == 17372
    np.testing.assert_almost_equal(
        aeri_ds['aeri_irt_equiv_temperature'].values[7], 286.081, decimal=3
    )
    np.testing.assert_almost_equal(
        aeri_ds['aeri_irt_equiv_temperature'].values[-10], 285.366, decimal=3
    )
    aeri_ds.close()
    del aeri_ds
