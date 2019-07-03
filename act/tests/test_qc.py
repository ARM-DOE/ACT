from act.io.armfiles import read_netcdf
from act.tests import EXAMPLE_IRT25m20s
import numpy as np


def test_qc_init():
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
