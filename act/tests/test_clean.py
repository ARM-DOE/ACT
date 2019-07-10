from act.io.armfiles import read_netcdf
from act.tests import EXAMPLE_CEIL1


def test_clean():
    ceil_ds = read_netcdf([EXAMPLE_CEIL1])
    ceil_ds.clean.cleanup(clean_arm_state_vars=['detection_status'])

    # Check that global attribures are removed
    global_attributes = ['qc_bit_comment',
                         'qc_bit_1_description',
                         'qc_bit_1_assessment',
                         'qc_bit_2_description',
                         'qc_bit_2_assessment'
                         'qc_bit_3_description',
                         'qc_bit_3_assessment',
                         'qc_bit_4_description',
                         'qc_bit_4_assessment']

    for glb_att in global_attributes:
        assert glb_att not in ceil_ds.attrs.keys()

    # Check that CF attributes are set including new flag_assessments
    assert 'flag_masks' in ceil_ds['qc_first_cbh'].attrs.keys()
    assert 'flag_meanings' in ceil_ds['qc_first_cbh'].attrs.keys()
    assert 'flag_assessments' in ceil_ds['qc_first_cbh'].attrs.keys()

    # Check the value of flag_assessments is as expected
    assert (all([ii == 'Bad' for ii in
            ceil_ds['qc_first_cbh'].attrs['flag_assessments']]))

    # Check the type is correct
    assert (type(ceil_ds['qc_first_cbh'].attrs['flag_masks'])) == list

    # Check that ancillary varibles is being added
    assert ('qc_first_cbh' in
            ceil_ds['first_cbh'].attrs['ancillary_variables'].split())

    # Check that state field is updated to CF
    assert 'flag_values' in ceil_ds['detection_status'].attrs.keys()
    assert 'flag_meanings' in ceil_ds['detection_status'].attrs.keys()
    assert ('detection_status' in
            ceil_ds['first_cbh'].attrs['ancillary_variables'].split())

    ceil_ds.close()
