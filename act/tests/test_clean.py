from act.io.armfiles import read_netcdf
from act.tests import EXAMPLE_CEIL1


def test_clean():
    # Read test data
    ceil_ds = read_netcdf([EXAMPLE_CEIL1])
    # Cleanup QC data
    ceil_ds.clean.cleanup(clean_arm_state_vars=['detection_status'])

    # Check that global attribures are removed
    global_attributes = ['qc_bit_comment',
                         'qc_bit_1_description',
                         'qc_bit_1_assessment',
                         'qc_bit_2_description',
                         'qc_bit_2_assessment'
                         'qc_bit_3_description',
                         'qc_bit_3_assessment'
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
    assert (ceil_ds['qc_first_cbh'].attrs['flag_meanings'] ==
            ['Value is equal to missing_value.',
             'Value is less than the fail_min.',
             'Value is greater than the fail_max.'])

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
    assert (ceil_ds['detection_status'].attrs['flag_meanings'] ==
            ['No significant backscatter',
             'One cloud base detected',
             'Two cloud bases detected',
             'Three cloud bases detected',
             'Full obscuration determined but no cloud base detected',
             'Some obscuration detected but determined to be transparent'])

    assert 'flag_0_description' not in ceil_ds['detection_status'].attrs.keys()
    assert ('detection_status' in
            ceil_ds['first_cbh'].attrs['ancillary_variables'].split())

    ceil_ds.close()
