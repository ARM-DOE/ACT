import numpy as np

import act


def test_calc_cog_sog():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_NAV)

    ds = act.utils.calc_cog_sog(ds)

    cog = ds['course_over_ground'].values
    sog = ds['speed_over_ground'].values

    np.testing.assert_almost_equal(cog[10], 170.987, decimal=3)
    np.testing.assert_almost_equal(sog[15], 0.448, decimal=3)

    ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    ds = act.utils.calc_cog_sog(ds)
    np.testing.assert_almost_equal(cog[10], 170.987, decimal=3)
    np.testing.assert_almost_equal(sog[15], 0.448, decimal=3)
