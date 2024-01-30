import numpy as np

from act.io.arm import read_arm_netcdf
from act.tests import EXAMPLE_CEIL1, EXAMPLE_CO2FLX4M, EXAMPLE_MET1


def test_global_qc_cleanup():
    ds = read_arm_netcdf(EXAMPLE_MET1)
    ds.load()
    ds.clean.cleanup()

    assert ds['qc_wdir_vec_mean'].attrs['flag_meanings'] == [
        'Value is equal to missing_value.',
        'Value is less than the fail_min.',
        'Value is greater than the fail_max.',
    ]
    assert ds['qc_wdir_vec_mean'].attrs['flag_masks'] == [1, 2, 4]
    assert ds['qc_wdir_vec_mean'].attrs['flag_assessments'] == [
        'Bad',
        'Bad',
        'Bad',
    ]

    assert ds['qc_temp_mean'].attrs['flag_meanings'] == [
        'Value is equal to missing_value.',
        'Value is less than the fail_min.',
        'Value is greater than the fail_max.',
        'Difference between current and previous values exceeds fail_delta.',
    ]
    assert ds['qc_temp_mean'].attrs['flag_masks'] == [1, 2, 4, 8]
    assert ds['qc_temp_mean'].attrs['flag_assessments'] == [
        'Bad',
        'Bad',
        'Bad',
        'Indeterminate',
    ]

    ds.close()
    del ds


def test_clean():
    # Read test data
    ceil_ds = read_arm_netcdf([EXAMPLE_CEIL1])
    # Cleanup QC data
    ceil_ds.clean.cleanup(clean_arm_state_vars=['detection_status'])

    # Check that global attribures are removed
    global_attributes = [
        'qc_bit_comment',
        'qc_bit_1_description',
        'qc_bit_1_assessment',
        'qc_bit_2_description',
        'qc_bit_2_assessment' 'qc_bit_3_description',
        'qc_bit_3_assessment',
    ]

    for glb_att in global_attributes:
        assert glb_att not in ceil_ds.attrs.keys()

    # Check that CF attributes are set including new flag_assessments
    var_name = 'qc_first_cbh'
    for attr_name in ['flag_masks', 'flag_meanings', 'flag_assessments']:
        assert attr_name in ceil_ds[var_name].attrs.keys()
        assert isinstance(ceil_ds[var_name].attrs[attr_name], list)

    # Check that the flag_mask values are set correctly
    assert ceil_ds['qc_first_cbh'].attrs['flag_masks'] == [1, 2, 4]

    # Check that the flag_meanings values are set correctly
    assert ceil_ds['qc_first_cbh'].attrs['flag_meanings'] == [
        'Value is equal to missing_value.',
        'Value is less than the fail_min.',
        'Value is greater than the fail_max.',
    ]

    # Check the value of flag_assessments is as expected
    assert ceil_ds['qc_first_cbh'].attrs['flag_assessments'] == ['Bad', 'Bad', 'Bad']

    # Check that ancillary varibles is being added
    assert 'qc_first_cbh' in ceil_ds['first_cbh'].attrs['ancillary_variables'].split()

    # Check that state field is updated to CF
    assert 'flag_values' in ceil_ds['detection_status'].attrs.keys()
    assert isinstance(ceil_ds['detection_status'].attrs['flag_values'], list)
    assert ceil_ds['detection_status'].attrs['flag_values'] == [0, 1, 2, 3, 4, 5]

    assert 'flag_meanings' in ceil_ds['detection_status'].attrs.keys()
    assert isinstance(ceil_ds['detection_status'].attrs['flag_meanings'], list)
    assert ceil_ds['detection_status'].attrs['flag_meanings'] == [
        'No significant backscatter',
        'One cloud base detected',
        'Two cloud bases detected',
        'Three cloud bases detected',
        'Full obscuration determined but no cloud base detected',
        'Some obscuration detected but determined to be transparent',
    ]

    assert 'flag_0_description' not in ceil_ds['detection_status'].attrs.keys()
    assert 'detection_status' in ceil_ds['first_cbh'].attrs['ancillary_variables'].split()

    ceil_ds.close()


def test_qc_remainder():
    ds = read_arm_netcdf(EXAMPLE_MET1)
    assert ds.clean.get_attr_info(variable='bad_name') is None
    del ds.attrs['qc_bit_comment']
    assert isinstance(ds.clean.get_attr_info(), dict)
    ds.attrs['qc_flag_comment'] = 'testing'
    ds.close()

    ds = read_arm_netcdf(EXAMPLE_MET1)
    ds.clean.cleanup(normalize_assessment=True)
    ds['qc_atmos_pressure'].attrs['units'] = 'testing'
    del ds['qc_temp_mean'].attrs['units']
    del ds['qc_temp_mean'].attrs['flag_masks']
    ds.clean.handle_missing_values()
    ds.close()

    ds = read_arm_netcdf(EXAMPLE_MET1)
    ds.attrs['qc_bit_1_comment'] = 'tesing'
    data = ds['qc_atmos_pressure'].values.astype(np.int64)
    data[0] = 2**32
    ds['qc_atmos_pressure'].values = data
    ds.clean.get_attr_info(variable='qc_atmos_pressure')
    ds.clean.clean_arm_state_variables('testname')
    ds.clean.cleanup()
    ds['qc_atmos_pressure'].attrs['standard_name'] = 'wrong_name'
    ds.clean.link_variables()
    assert ds['qc_atmos_pressure'].attrs['standard_name'] == 'quality_flag'
    ds.close()


def test_qc_flag_description():
    """
    This will check if the cleanup() method will correctly convert convert
    flag_#_description to CF flag_masks and flag_meanings.

    """

    ds = read_arm_netcdf(EXAMPLE_CO2FLX4M)
    ds.clean.cleanup()
    qc_var_name = ds.qcfilter.check_for_ancillary_qc(
        'momentum_flux', add_if_missing=False, cleanup=False
    )

    assert isinstance(ds[qc_var_name].attrs['flag_masks'], list)
    assert isinstance(ds[qc_var_name].attrs['flag_meanings'], list)
    assert isinstance(ds[qc_var_name].attrs['flag_assessments'], list)
    assert ds[qc_var_name].attrs['standard_name'] == 'quality_flag'

    assert len(ds[qc_var_name].attrs['flag_masks']) == 9
    unique_flag_assessments = list({'Acceptable', 'Indeterminate', 'Bad'})
    for f in list(set(ds[qc_var_name].attrs['flag_assessments'])):
        assert f in unique_flag_assessments
