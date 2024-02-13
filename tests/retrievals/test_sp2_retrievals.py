import numpy as np
import pytest

import act

try:
    import pysp2  # noqa

    PYSP2_AVAILABLE = True
except ImportError:
    PYSP2_AVAILABLE = False


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason='PySP2 is not installed.')
def test_sp2_waveform_stats():
    my_sp2b = act.io.read_sp2(act.tests.EXAMPLE_SP2B)
    my_ini = act.tests.EXAMPLE_INI
    my_binary = act.qc.get_waveform_statistics(my_sp2b, my_ini, parallel=False)
    assert my_binary.PkHt_ch1.max() == 62669.4
    np.testing.assert_almost_equal(np.nanmax(my_binary.PkHt_ch0.values), 98708.92915295, decimal=1)
    np.testing.assert_almost_equal(np.nanmax(my_binary.PkHt_ch4.values), 65088.39598033, decimal=1)


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason='PySP2 is not installed.')
def test_sp2_psds():
    my_sp2b = act.io.read_sp2(act.tests.EXAMPLE_SP2B)
    my_ini = act.tests.EXAMPLE_INI
    my_binary = act.qc.get_waveform_statistics(my_sp2b, my_ini, parallel=False)
    my_hk = act.io.read_hk_file(act.tests.EXAMPLE_HK)
    my_binary = act.retrievals.calc_sp2_diams_masses(my_binary)
    scatrejectkey = my_binary['ScatRejectKey'].values
    assert np.nanmax(my_binary['ScatDiaBC50'].values[scatrejectkey == 0]) < 1000.0
    my_psds = act.retrievals.process_sp2_psds(my_binary, my_hk, my_ini)
    np.testing.assert_almost_equal(my_psds['NumConcIncan'].max(), 0.95805343)
