import numpy as np

import act


def test_planck_converter():
    wnum = 1100
    temp = 300
    radiance = 81.5
    result = act.utils.radiance_utils.planck_converter(wnum=wnum, temperature=temp)
    np.testing.assert_almost_equal(result, radiance, decimal=1)
    result = act.utils.radiance_utils.planck_converter(wnum=wnum, radiance=radiance)
    assert np.ceil(result) == temp
    np.testing.assert_raises(ValueError, act.utils.radiance_utils.planck_converter)
