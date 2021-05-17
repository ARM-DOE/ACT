from act.io.armfiles import read_netcdf
from act.tests import (EXAMPLE_IRT25m20s, EXAMPLE_METE40, EXAMPLE_CEIL1,
                       EXAMPLE_MFRSR, EXAMPLE_MET1, EXAMPLE_CO2FLX4M)
from act.qc.arm import add_dqr_to_qc
from act.qc.radiometer_tests import fft_shading_test
from act.qc.qcfilter import parse_bit, set_bit, unset_bit
import numpy as np
import pytest
import copy


def test_fft_shading_test():
    obj = read_netcdf(EXAMPLE_MFRSR)
    obj.clean.cleanup()
    obj = fft_shading_test(obj)
    qc_data = obj['qc_diffuse_hemisp_narrowband_filter4']
    assert np.nansum(qc_data.values) == 456


def test_qc_test_errors():
    ds_object = read_netcdf(EXAMPLE_MET1)
    var_name = 'temp_mean'

    assert ds_object.qcfilter.add_less_test(var_name, None) is None
    assert ds_object.qcfilter.add_greater_test(var_name, None) is None
    assert ds_object.qcfilter.add_less_equal_test(var_name, None) is None
    assert ds_object.qcfilter.add_equal_to_test(var_name, None) is None
    assert ds_object.qcfilter.add_not_equal_to_test(var_name, None) is None


def test_arm_qc():
    # Test DQR Webservice using known DQR
    variable = 'wspd_vec_mean'
    qc_variable = 'qc_' + variable
    obj = read_netcdf(EXAMPLE_METE40)

    # DQR webservice does go down, so ensure it
    # properly runs first before testing
    try:
        obj = add_dqr_to_qc(obj, variable=variable)
        ran = True
        obj.attrs['_datastream'] = obj.attrs['datastream']
        del obj.attrs['datastream']
        obj2 = add_dqr_to_qc(obj, variable=variable)
        with np.testing.assert_raises(ValueError):
            del obj.attrs['_datastream']
            add_dqr_to_qc(obj, variable=variable)

        obj4 = add_dqr_to_qc(obj)
    except ValueError:
        ran = False

    if ran:
        assert qc_variable in obj
        dqr = [True for d in obj[qc_variable].attrs['flag_meanings'] if 'D190529.4' in d]
        assert dqr[0] is True
        assert 'Suspect' not in obj[qc_variable].attrs['flag_assessments']
        assert 'Incorrect' not in obj[qc_variable].attrs['flag_assessments']

        assert qc_variable in obj2
        dqr = [True for d in obj2[qc_variable].attrs['flag_meanings'] if 'D190529.4' in d]
        assert dqr[0] is True
        assert 'Suspect' not in obj2[qc_variable].attrs['flag_assessments']
        assert 'Incorrect' not in obj2[qc_variable].attrs['flag_assessments']

        assert qc_variable in obj4
        dqr = [True for d in obj4[qc_variable].attrs['flag_meanings'] if 'D190529.4' in d]
        assert dqr[0] is True
        assert 'Suspect' not in obj4[qc_variable].attrs['flag_assessments']
        assert 'Incorrect' not in obj4[qc_variable].attrs['flag_assessments']


def test_qcfilter():
    ds_object = read_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'
    expected_qc_var_name = 'qc_' + var_name

    # Perform adding of quality control variables to object
    result = ds_object.qcfilter.add_test(var_name, test_meaning='Birds!')
    assert isinstance(result, dict)
    qc_var_name = result['qc_variable_name']
    assert qc_var_name == expected_qc_var_name

    # Check that new linking and describing attributes are set
    assert ds_object[qc_var_name].attrs['standard_name'] == 'quality_flag'
    assert ds_object[var_name].attrs['ancillary_variables'] == qc_var_name

    # Check that CF attributes are set including new flag_assessments
    assert 'flag_masks' in ds_object[qc_var_name].attrs.keys()
    assert 'flag_meanings' in ds_object[qc_var_name].attrs.keys()
    assert 'flag_assessments' in ds_object[qc_var_name].attrs.keys()

    # Check that the values of the attributes are set correctly
    assert ds_object[qc_var_name].attrs['flag_assessments'][0] == 'Bad'
    assert ds_object[qc_var_name].attrs['flag_meanings'][0] == 'Birds!'
    assert ds_object[qc_var_name].attrs['flag_masks'][0] == 1

    # Set some test values
    index = [0, 1, 2, 30]
    ds_object.qcfilter.set_test(var_name, index=index,
                                test_number=result['test_number'])

    # Add a new test and set values
    index2 = [6, 7, 8, 50]
    ds_object.qcfilter.add_test(var_name, index=index2,
                                test_number=9,
                                test_meaning='testing high number',
                                test_assessment='Suspect')

    # Retrieve data from object as numpy masked array. Count number of masked
    # elements and ensure equal to size of index array.
    data = ds_object.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
    assert np.ma.count_masked(data) == len(index)

    data = ds_object.qcfilter.get_masked_data(
        var_name, rm_assessments='Suspect', return_nan_array=True)
    assert np.sum(np.isnan(data)) == len(index2)

    data = ds_object.qcfilter.get_masked_data(
        var_name, rm_assessments=['Bad', 'Suspect'], ma_fill_value=np.nan)
    assert np.ma.count_masked(data) == len(index + index2)

    # Test internal function for returning the index array of where the
    # tests are set.
    assert np.sum(ds_object.qcfilter.get_qc_test_mask(
        var_name, result['test_number'], return_index=True) -
        np.array(index, dtype=np.int)) == 0

    # Unset a test
    ds_object.qcfilter.unset_test(var_name, index=0,
                                  test_number=result['test_number'])
    # Remove the test
    ds_object.qcfilter.remove_test(var_name,
                                   test_number=result['test_number'])
    pytest.raises(ValueError, ds_object.qcfilter.add_test, var_name)
    pytest.raises(ValueError, ds_object.qcfilter.remove_test, var_name)

    ds_object.close()

    pytest.raises(ValueError, parse_bit, [1, 2])
    pytest.raises(ValueError, parse_bit, -1)

    assert set_bit(0, 16) == 32768
    data = range(0, 4)
    assert isinstance(set_bit(list(data), 2), list)
    assert isinstance(set_bit(tuple(data), 2), tuple)
    assert isinstance(unset_bit(list(data), 2), list)
    assert isinstance(unset_bit(tuple(data), 2), tuple)

    # Fill in missing tests
    ds_object = read_netcdf(EXAMPLE_IRT25m20s)
    del ds_object[var_name].attrs['long_name']
    # Test creating a qc variable
    ds_object.qcfilter.create_qc_variable(var_name)
    # Test creating a second qc variable and of flag type
    ds_object.qcfilter.create_qc_variable(var_name, flag_type=True)
    result = ds_object.qcfilter.add_test(var_name, index=[1, 2, 3],
                                         test_number=9,
                                         test_meaning='testing high number',
                                         flag_value=True)
    ds_object.qcfilter.set_test(var_name, index=5, test_number=9, flag_value=True)
    data = ds_object.qcfilter.get_masked_data(var_name)
    assert np.isclose(np.sum(data), 42674.766, 0.01)
    data = ds_object.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
    assert np.isclose(np.sum(data), 42643.195, 0.01)

    ds_object.qcfilter.unset_test(var_name, test_number=9, flag_value=True)
    ds_object.qcfilter.unset_test(var_name, index=1, test_number=9, flag_value=True)
    assert ds_object.qcfilter.available_bit(result['qc_variable_name']) == 10
    assert ds_object.qcfilter.available_bit(result['qc_variable_name'], recycle=True) == 1
    ds_object.qcfilter.remove_test(var_name, test_number=9, flag_value=True)

    ds_object.qcfilter.update_ancillary_variable(var_name)
    # Test updating ancillary variable if does not exist
    ds_object.qcfilter.update_ancillary_variable('not_a_variable_name')
    # Change ancillary_variables attribute to test if add correct qc variable correctly
    ds_object[var_name].attrs['ancillary_variables'] = 'a_different_name'
    ds_object.qcfilter.update_ancillary_variable(var_name,
                                                 qc_var_name=expected_qc_var_name)
    assert (expected_qc_var_name in
            ds_object[var_name].attrs['ancillary_variables'])

    # Test flag QC
    var_name = 'inst_sfc_ir_temp'
    qc_var_name = 'qc_' + var_name
    ds_object.qcfilter.create_qc_variable(var_name, flag_type=True)
    assert qc_var_name in list(ds_object.data_vars)
    assert 'flag_values' in ds_object[qc_var_name].attrs.keys()
    assert 'flag_masks' not in ds_object[qc_var_name].attrs.keys()
    del ds_object[qc_var_name]

    qc_var_name = ds_object.qcfilter.check_for_ancillary_qc(
        var_name, add_if_missing=True, cleanup=False, flag_type=True)
    assert qc_var_name in list(ds_object.data_vars)
    assert 'flag_values' in ds_object[qc_var_name].attrs.keys()
    assert 'flag_masks' not in ds_object[qc_var_name].attrs.keys()
    del ds_object[qc_var_name]

    ds_object.qcfilter.add_missing_value_test(var_name, flag_value=True, prepend_text='arm')
    ds_object.qcfilter.add_test(var_name, index=list(range(0, 20)), test_number=2,
                                test_meaning='Testing flag', flag_value=True,
                                test_assessment='Suspect')
    assert qc_var_name in list(ds_object.data_vars)
    assert 'flag_values' in ds_object[qc_var_name].attrs.keys()
    assert 'flag_masks' not in ds_object[qc_var_name].attrs.keys()
    assert 'standard_name' in ds_object[qc_var_name].attrs.keys()
    assert ds_object[qc_var_name].attrs['flag_values'] == [1, 2]
    assert ds_object[qc_var_name].attrs['flag_assessments'] == ['Bad', 'Suspect']

    ds_object.close()


def test_qctests():
    ds_object = read_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'

    # Add in one missing value and test for that missing value
    data = ds_object[var_name].values
    data[0] = np.nan
    ds_object[var_name].values = data
    result = ds_object.qcfilter.add_missing_value_test(var_name)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert data.mask[0]

    # less than min test
    limit_value = 6.8
    result = ds_object.qcfilter.add_less_test(var_name, limit_value, prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 54
    assert 'fail_min' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_min'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_min'], limit_value)

    result = ds_object.qcfilter.add_less_test(var_name, limit_value, test_assessment='Suspect')
    assert 'warn_min' in ds_object[result['qc_variable_name']].attrs.keys()

    # greator than max test
    limit_value = 12.7
    result = ds_object.qcfilter.add_greater_test(var_name, limit_value, prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 61
    assert 'fail_max' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_max'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_max'], limit_value)

    result = ds_object.qcfilter.add_greater_test(var_name, limit_value, test_assessment='Suspect')
    assert 'warn_max' in ds_object[result['qc_variable_name']].attrs.keys()

    # less than or equal test
    limit_value = 6.9
    result = ds_object.qcfilter.add_less_equal_test(var_name, limit_value,
                                                    test_assessment='Suspect',
                                                    prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 149
    assert 'warn_min' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_min'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_min'], limit_value)

    result = ds_object.qcfilter.add_less_equal_test(var_name, limit_value)
    assert 'fail_min' in ds_object[result['qc_variable_name']].attrs.keys()

    # greater than or equal test
    limit_value = 12
    result = ds_object.qcfilter.add_greater_equal_test(var_name, limit_value,
                                                       test_assessment='Suspect',
                                                       prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 606
    assert 'warn_max' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_max'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_max'], limit_value)

    result = ds_object.qcfilter.add_greater_equal_test(var_name, limit_value)
    assert 'fail_max' in ds_object[result['qc_variable_name']].attrs.keys()

    # equal to test
    limit_value = 7.6705
    result = ds_object.qcfilter.add_equal_to_test(var_name, limit_value, prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 2
    assert 'fail_equal_to' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_equal_to'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_equal_to'], limit_value)

    result = ds_object.qcfilter.add_equal_to_test(var_name, limit_value,
                                                  test_assessment='Indeterminate')
    assert 'warn_equal_to' in ds_object[result['qc_variable_name']].attrs.keys()

    # not equal to test
    limit_value = 7.6705
    result = ds_object.qcfilter.add_not_equal_to_test(var_name, limit_value,
                                                      test_assessment='Indeterminate',
                                                      prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 4318
    assert 'warn_not_equal_to' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_not_equal_to'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_not_equal_to'], limit_value)

    result = ds_object.qcfilter.add_not_equal_to_test(var_name, limit_value)
    assert 'fail_not_equal_to' in ds_object[result['qc_variable_name']].attrs.keys()

    # outside range test
    limit_value1 = 6.8
    limit_value2 = 12.7
    result = ds_object.qcfilter.add_outside_test(var_name, limit_value1, limit_value2,
                                                 prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 115
    assert 'fail_lower_range' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_lower_range'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_lower_range'], limit_value1)
    assert 'fail_upper_range' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_upper_range'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_upper_range'], limit_value2)

    result = ds_object.qcfilter.add_outside_test(var_name, limit_value1, limit_value2,
                                                 test_assessment='Indeterminate')
    assert 'warn_lower_range' in ds_object[result['qc_variable_name']].attrs.keys()
    assert 'warn_upper_range' in ds_object[result['qc_variable_name']].attrs.keys()

    # inside range test
    limit_value1 = 7
    limit_value2 = 8
    result = ds_object.qcfilter.add_inside_test(var_name, limit_value1, limit_value2,
                                                prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 479
    assert 'fail_lower_range_inner' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_lower_range_inner'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_lower_range_inner'],
                      limit_value1)
    assert 'fail_upper_range_inner' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_upper_range_inner'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_upper_range_inner'],
                      limit_value2)

    result = ds_object.qcfilter.add_inside_test(var_name, limit_value1, limit_value2,
                                                test_assessment='Indeterminate')
    assert 'warn_lower_range_inner' in ds_object[result['qc_variable_name']].attrs.keys()
    assert 'warn_upper_range_inner' in ds_object[result['qc_variable_name']].attrs.keys()

    # delta test
    test_limit = 0.05
    result = ds_object.qcfilter.add_delta_test(var_name, test_limit, prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 175
    assert 'warn_delta' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_delta'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_delta'], test_limit)

    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_assessments=['Suspect', 'Bad'])
    assert np.ma.count_masked(data) == 4320

    result = ds_object.qcfilter.add_delta_test(var_name, test_limit, test_assessment='Bad')
    assert 'fail_delta' in ds_object[result['qc_variable_name']].attrs.keys()

    comp_object = read_netcdf(EXAMPLE_IRT25m20s)
    result = ds_object.qcfilter.add_difference_test(
        var_name, {comp_object.attrs['datastream']: comp_object},
        var_name, diff_limit=1, prepend_text='arm')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert not (data.mask).all()

    comp_object.close()
    ds_object.close()


def test_qctests_dos():
    ds_object = read_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'

    # persistence test
    data = ds_object[var_name].values
    data[1000:2500] = data[1000]
    ds_object[var_name].values = data
    ds_object.qcfilter.add_persistence_test(var_name)
    qc_var_name = ds_object.qcfilter.check_for_ancillary_qc(
        var_name, add_if_missing=False, cleanup=False, flag_type=False)
    test_meaning = ('Data failing persistence test. Standard Deviation over a '
                    'window of 10 values less than 0.0001.')
    assert ds_object[qc_var_name].attrs['flag_meanings'][-1] == test_meaning
    assert np.sum(ds_object[qc_var_name].values) == 1500

    ds_object.qcfilter.add_persistence_test(var_name, window=10000, prepend_text='DQO')
    test_meaning = ('DQO: Data failing persistence test. Standard Deviation over a window of '
                    '4320 values less than 0.0001.')
    assert ds_object[qc_var_name].attrs['flag_meanings'][-1] == test_meaning


def test_datafilter():
    ds = read_netcdf(EXAMPLE_MET1)
    ds.clean.cleanup()

    var_name = 'atmos_pressure'

    ds_1 = ds.mean()

    ds.qcfilter.add_less_test(var_name, 99, test_assessment='Bad')
    ds.qcfilter.datafilter(rm_assessments='Bad')
    ds_2 = ds.mean()

    assert np.isclose(ds_1[var_name].values, 98.86, atol=0.01)
    assert np.isclose(ds_2[var_name].values, 99.15, atol=0.01)

    ds.close()


def test_qc_remainder():
    ds = read_netcdf(EXAMPLE_MET1)
    assert ds.clean.get_attr_info(variable='bad_name') is None
    del ds.attrs['qc_bit_comment']
    assert isinstance(ds.clean.get_attr_info(), dict)
    ds.attrs['qc_flag_comment'] = 'testing'
    ds.close()

    ds = read_netcdf(EXAMPLE_MET1)
    ds.clean.cleanup(normalize_assessment=True)
    ds['qc_atmos_pressure'].attrs['units'] = 'testing'
    del ds['qc_temp_mean'].attrs['units']
    del ds['qc_temp_mean'].attrs['flag_masks']
    ds.clean.handle_missing_values()
    ds.close()

    ds = read_netcdf(EXAMPLE_MET1)
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

    ds = read_netcdf(EXAMPLE_CO2FLX4M)
    ds.clean.cleanup()
    qc_var_name = ds.qcfilter.check_for_ancillary_qc('momentum_flux', add_if_missing=False,
                                                     cleanup=False)

    assert isinstance(ds[qc_var_name].attrs['flag_masks'], list)
    assert isinstance(ds[qc_var_name].attrs['flag_meanings'], list)
    assert isinstance(ds[qc_var_name].attrs['flag_assessments'], list)
    assert ds[qc_var_name].attrs['standard_name'] == 'quality_flag'

    assert len(ds[qc_var_name].attrs['flag_masks']) == 9
    unique_flag_assessments = list(set(['Acceptable', 'Indeterminate', 'Bad']))
    assert list(set(ds[qc_var_name].attrs['flag_assessments'])) == unique_flag_assessments


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


def test_compare_time_series_trends():

    drop_vars = ['base_time', 'time_offset', 'atmos_pressure', 'qc_atmos_pressure',
                 'temp_std', 'rh_mean', 'qc_rh_mean', 'rh_std', 'vapor_pressure_mean',
                 'qc_vapor_pressure_mean', 'vapor_pressure_std', 'wspd_arith_mean',
                 'qc_wspd_arith_mean', 'wspd_vec_mean', 'qc_wspd_vec_mean', 'wdir_vec_mean',
                 'qc_wdir_vec_mean', 'wdir_vec_std', 'tbrg_precip_total', 'qc_tbrg_precip_total',
                 'tbrg_precip_total_corr', 'qc_tbrg_precip_total_corr', 'org_precip_rate_mean',
                 'qc_org_precip_rate_mean', 'pwd_err_code', 'pwd_mean_vis_1min', 'qc_pwd_mean_vis_1min',
                 'pwd_mean_vis_10min', 'qc_pwd_mean_vis_10min', 'pwd_pw_code_inst',
                 'qc_pwd_pw_code_inst', 'pwd_pw_code_15min', 'qc_pwd_pw_code_15min',
                 'pwd_pw_code_1hr', 'qc_pwd_pw_code_1hr', 'pwd_precip_rate_mean_1min',
                 'qc_pwd_precip_rate_mean_1min', 'pwd_cumul_rain', 'qc_pwd_cumul_rain',
                 'pwd_cumul_snow', 'qc_pwd_cumul_snow', 'logger_volt', 'qc_logger_volt',
                 'logger_temp', 'qc_logger_temp', 'lat', 'lon', 'alt']
    ds = read_netcdf(EXAMPLE_MET1, drop_variables=drop_vars)
    ds.clean.cleanup()
    ds2 = copy.deepcopy(ds)

    var_name = 'temp_mean'
    qc_var_name = ds.qcfilter.check_for_ancillary_qc(var_name, add_if_missing=False,
                                                     cleanup=False, flag_type=False)
    ds.qcfilter.compare_time_series_trends(var_name=var_name, time_shift=60,
                                           comp_var_name=var_name, comp_dataset=ds2,
                                           time_qc_threshold=60 * 10)

    test_description = ('Time shift detected with Minimum Difference test. Comparison of '
                        'temp_mean with temp_mean off by 0 seconds exceeding absolute '
                        'threshold of 600 seconds.')
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_description

    time = ds2['time'].values + np.timedelta64(1, 'h')
    time_attrs = ds2['time'].attrs
    ds2 = ds2.assign_coords({'time': time})
    ds2['time'].attrs = time_attrs

    ds.qcfilter.compare_time_series_trends(var_name=var_name, comp_dataset=ds2, time_step=60,
                                           time_match_threshhold=50)

    test_description = ('Time shift detected with Minimum Difference test. Comparison of '
                        'temp_mean with temp_mean off by 3600 seconds exceeding absolute '
                        'threshold of 900 seconds.')
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_description
