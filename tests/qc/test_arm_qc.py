import numpy as np

from act.io.arm import read_arm_netcdf
from act.qc.arm import add_dqr_to_qc, print_dqr
from act.tests import EXAMPLE_ENA_MET, EXAMPLE_OLD_QC


def test_scalar_dqr():
    # Test DQR Webservice using known DQR
    ds = read_arm_netcdf(EXAMPLE_ENA_MET)

    # DQR webservice does go down, so ensure it
    # properly runs first before testing
    try:
        ds = add_dqr_to_qc(ds, assessment='Reprocessed,Suspect,Incorrect')
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


def test_print_dqr():
    dqr = print_dqr('sgpmetE13.b1', '2023-01-01', '2024-01-01', variable='pwd_cumul_rain')
    assert 'D231114.33' in dqr
    assert 'pwd_cumul_rain' in dqr['D231114.33']['variables']

    with np.testing.assert_raises(ValueError):
        dqr = print_dqr('spmetE13.b1', '2023-01-01', '2024-01-01', variable='pwd_cumul_rain')
