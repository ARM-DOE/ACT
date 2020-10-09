from act.io.armfiles import read_netcdf
from act.tests import (EXAMPLE_IRT25m20s, EXAMPLE_METE40,
                       EXAMPLE_MFRSR, EXAMPLE_MET1, EXAMPLE_CO2FLX4M)
from act.qc.arm import add_dqr_to_qc
from act.qc.radiometer_tests import fft_shading_test
from act.qc.qcfilter import parse_bit, set_bit, unset_bit
import numpy as np
import pytest


def test_fft_shading_test():
    obj = read_netcdf(EXAMPLE_MFRSR)
    obj.clean.cleanup()
    obj = fft_shading_test(obj)
    qc_data = obj['qc_diffuse_hemisp_narrowband_filter4']
    assert np.nansum(qc_data.values) == 456


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
    except ValueError:
        ran = False

    if ran:
        assert qc_variable in obj
        dqr = [True for d in obj[qc_variable].attrs['flag_meanings'] if 'D190529.4' in d]
        assert dqr[0] is True

        assert 'Suspect' not in obj[qc_variable].attrs['flag_assessments']
        assert 'Incorrect' not in obj[qc_variable].attrs['flag_assessments']


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

    ds_object.qcfilter.add_missing_value_test(var_name, flag_value=True)
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
    result = ds_object.qcfilter.add_less_test(var_name, limit_value)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 54
    assert 'fail_min' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_min'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_min'], limit_value)

    # greator than max test
    limit_value = 12.7
    result = ds_object.qcfilter.add_greater_test(var_name, limit_value)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 61
    assert 'fail_max' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_max'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_max'], limit_value)

    # less than or equal test
    limit_value = 6.9
    result = ds_object.qcfilter.add_less_equal_test(var_name, limit_value,
                                                    test_assessment='Suspect')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 149
    assert 'warn_min' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_min'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_min'], limit_value)

    # greater than or equal test
    limit_value = 12
    result = ds_object.qcfilter.add_greater_equal_test(var_name, limit_value,
                                                       test_assessment='Suspect')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 606
    assert 'warn_max' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_max'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_max'], limit_value)

    # equal to test
    limit_value = 7.6705
    result = ds_object.qcfilter.add_equal_to_test(var_name, limit_value)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 2
    assert 'fail_equal_to' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_equal_to'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_equal_to'], limit_value)

    # not equal to test
    limit_value = 7.6705
    result = ds_object.qcfilter.add_not_equal_to_test(var_name, limit_value,
                                                      test_assessment='Indeterminate')
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 4318
    assert 'warn_not_equal_to' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_not_equal_to'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_not_equal_to'], limit_value)

    # outside range test
    limit_value1 = 6.8
    limit_value2 = 12.7
    result = ds_object.qcfilter.add_outside_test(var_name, limit_value1, limit_value2)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 115
    assert 'fail_lower_range' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_lower_range'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_lower_range'], limit_value1)
    assert 'fail_upper_range' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['fail_upper_range'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['fail_upper_range'], limit_value2)

    # inside range test
    limit_value1 = 7
    limit_value2 = 8
    result = ds_object.qcfilter.add_inside_test(var_name, limit_value1, limit_value2)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
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

    # delta test
    test_limit = 0.05
    result = ds_object.qcfilter.add_delta_test(var_name, test_limit)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert np.ma.count_masked(data) == 175
    assert 'warn_delta' in ds_object[result['qc_variable_name']].attrs.keys()
    assert (ds_object[result['qc_variable_name']].attrs['warn_delta'].dtype ==
            ds_object[result['variable_name']].values.dtype)
    assert np.isclose(ds_object[result['qc_variable_name']].attrs['warn_delta'], test_limit)

    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_assessments=['Suspect', 'Bad'])
    assert np.ma.count_masked(data) == 1235

    comp_object = read_netcdf(EXAMPLE_IRT25m20s)
    result = ds_object.qcfilter.add_difference_test(
        var_name, {comp_object.attrs['datastream']: comp_object},
        var_name, diff_limit=1)
    data = ds_object.qcfilter.get_masked_data(var_name,
                                              rm_tests=result['test_number'])
    assert not (data.mask).all()

    comp_object.close()
    ds_object.close()


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
