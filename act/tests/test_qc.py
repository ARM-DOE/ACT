from act.io.armfiles import read_netcdf
from act.tests import EXAMPLE_IRT25m20s, EXAMPLE_METE40, EXAMPLE_MFRSR
from act.qc.arm import add_dqr_to_qc
from act.qc.radiometer_tests import fft_shading_test
import numpy as np


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
