import numpy as np
import pytest

import act


@pytest.mark.skipif(not act.io.icartt._ICARTT_AVAILABLE, reason='ICARTT is not installed.')
def test_read_icartt():
    result = act.io.icartt.read_icartt(act.tests.EXAMPLE_AAF_ICARTT)
    assert 'pitch' in result
    assert len(result['time'].values) == 14087
    assert result['true_airspeed'].units == 'm/s'
    assert 'Revision' in result.attrs
    np.testing.assert_almost_equal(result['static_pressure'].mean(), 708.75, decimal=2)
