import numpy as np

from act.io.arm import read_arm_netcdf
from act.qc.arm import add_dqr_to_qc
from act.tests import EXAMPLE_ENA_MET, EXAMPLE_OLD_QC


def test_scalar_dqr():
    # Test DQR Webservice using known DQR
    ds = read_arm_netcdf(EXAMPLE_ENA_MET)

    # DQR webservice does go down, so ensure it
    # properly runs first before testing
    try:
        ds = add_dqr_to_qc(ds)
        ran = True
    except ValueError:
        ran = False

    if ran:
        assert 'qc_lat' in ds
        assert np.size(ds['qc_lon'].values) == 1
        assert np.size(ds['qc_lat'].values) == 1
        assert np.size(ds['qc_alt'].values) == 1
        assert np.size(ds['base_time'].values) == 1


def test_get_attr_info():
    ds = read_arm_netcdf(EXAMPLE_OLD_QC, cleanup_qc=True)
    assert 'flag_assessments' in ds['qc_lv'].attrs
    assert 'fail_min' in ds['qc_lv'].attrs
    assert ds['qc_lv'].attrs['flag_assessments'][0] == 'Bad'
    assert ds['qc_lv'].attrs['flag_masks'][-1] == 4
