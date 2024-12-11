import dask.array as da
import numpy as np

from act.io.arm import read_arm_netcdf
from act.tests import EXAMPLE_MET1, EXAMPLE_IRT25m20s


def test_qctests():
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'

    # Add in one missing value and test for that missing value
    data = ds[var_name].values
    data[0] = np.nan
    ds[var_name].data = da.from_array(data)
    result = ds.qcfilter.add_missing_value_test(var_name)
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert data.mask[0]

    result = ds.qcfilter.add_missing_value_test(var_name, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert data == np.array([0])
    ds.qcfilter.remove_test(var_name, test_number=result['test_number'])

    # less than min test
    limit_value = 6.8
    result = ds.qcfilter.add_less_test(
        var_name, limit_value, prepend_text='arm', limit_attr_name='fail_min'
    )

    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 54
    assert 'fail_min' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['fail_min'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['fail_min'], limit_value)

    result = ds.qcfilter.add_less_test(var_name, limit_value, test_assessment='Suspect')
    assert 'warn_min' in ds[result['qc_variable_name']].attrs.keys()

    limit_value = 8
    result = ds.qcfilter.add_less_test(var_name, limit_value)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 2911939
    result = ds.qcfilter.add_less_test(var_name, limit_value, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 2911939

    # greator than max test
    limit_value = 12.7
    result = ds.qcfilter.add_greater_test(
        var_name, limit_value, prepend_text='arm', limit_attr_name='fail_max'
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 61
    assert 'fail_max' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['fail_max'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['fail_max'], limit_value)

    result = ds.qcfilter.add_greater_test(var_name, limit_value, test_assessment='Suspect')
    assert 'warn_max' in ds[result['qc_variable_name']].attrs.keys()

    result = ds.qcfilter.add_greater_test(var_name, limit_value, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 125458
    result = ds.qcfilter.add_greater_test(var_name, limit_value)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 125458

    # less than or equal test
    limit_value = 6.9
    result = ds.qcfilter.add_less_equal_test(
        var_name,
        limit_value,
        test_assessment='Suspect',
        prepend_text='arm',
        limit_attr_name='warn_min',
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 149
    assert 'warn_min' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['warn_min'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['warn_min'], limit_value)

    result = ds.qcfilter.add_less_equal_test(var_name, limit_value)
    assert 'fail_min' in ds[result['qc_variable_name']].attrs.keys()

    result = ds.qcfilter.add_less_equal_test(var_name, limit_value, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 601581
    result = ds.qcfilter.add_less_equal_test(var_name, limit_value)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 601581

    # greater than or equal test
    result = ds.qcfilter.add_greater_equal_test(var_name, None)
    limit_value = 12
    result = ds.qcfilter.add_greater_equal_test(
        var_name,
        limit_value,
        test_assessment='Suspect',
        prepend_text='arm',
        limit_attr_name='warn_max',
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 606
    assert 'warn_max' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['warn_max'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['warn_max'], limit_value)

    result = ds.qcfilter.add_greater_equal_test(var_name, limit_value)
    assert 'fail_max' in ds[result['qc_variable_name']].attrs.keys()

    result = ds.qcfilter.add_greater_equal_test(var_name, limit_value, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 1189873
    result = ds.qcfilter.add_greater_equal_test(var_name, limit_value)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 1189873

    # equal to test
    limit_value = 7.6705
    result = ds.qcfilter.add_equal_to_test(
        var_name, limit_value, prepend_text='arm', limit_attr_name='fail_equal_to'
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 2
    assert 'fail_equal_to' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['fail_equal_to'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['fail_equal_to'], limit_value)

    result = ds.qcfilter.add_equal_to_test(var_name, limit_value, test_assessment='Indeterminate')
    assert 'warn_equal_to' in ds[result['qc_variable_name']].attrs.keys()

    result = ds.qcfilter.add_equal_to_test(var_name, limit_value, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 8631
    result = ds.qcfilter.add_equal_to_test(var_name, limit_value)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 8631

    # not equal to test
    limit_value = 7.6705
    result = ds.qcfilter.add_not_equal_to_test(
        var_name,
        limit_value,
        test_assessment='Indeterminate',
        prepend_text='arm',
        limit_attr_name='warn_not_equal_to',
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 4318
    assert 'warn_not_equal_to' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['warn_not_equal_to'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['warn_not_equal_to'], limit_value)

    result = ds.qcfilter.add_not_equal_to_test(var_name, limit_value)
    assert 'fail_not_equal_to' in ds[result['qc_variable_name']].attrs.keys()

    result = ds.qcfilter.add_not_equal_to_test(var_name, limit_value, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 9320409
    result = ds.qcfilter.add_not_equal_to_test(var_name, limit_value)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 9320409

    # outside range test
    limit_value1 = 6.8
    limit_value2 = 12.7
    result = ds.qcfilter.add_outside_test(
        var_name,
        limit_value1,
        limit_value2,
        prepend_text='arm',
        limit_attr_names=['fail_lower_range', 'fail_upper_range'],
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 115
    assert 'fail_lower_range' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['fail_lower_range'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['fail_lower_range'], limit_value1)
    assert 'fail_upper_range' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['fail_upper_range'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['fail_upper_range'], limit_value2)

    result = ds.qcfilter.add_outside_test(
        var_name, limit_value1, limit_value2, test_assessment='Indeterminate'
    )
    assert 'warn_lower_range' in ds[result['qc_variable_name']].attrs.keys()
    assert 'warn_upper_range' in ds[result['qc_variable_name']].attrs.keys()

    result = ds.qcfilter.add_outside_test(var_name, limit_value1, limit_value2, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 342254
    result = ds.qcfilter.add_outside_test(
        var_name,
        limit_value1,
        limit_value2,
    )
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 342254

    # Starting to run out of space for tests. Remove some tests.
    for ii in range(16, 30):
        ds.qcfilter.remove_test(var_name, test_number=ii)

    # inside range test
    limit_value1 = 7
    limit_value2 = 8
    result = ds.qcfilter.add_inside_test(
        var_name,
        limit_value1,
        limit_value2,
        prepend_text='arm',
        limit_attr_names=['fail_lower_range_inner', 'fail_upper_range_inner'],
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 479
    assert 'fail_lower_range_inner' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['fail_lower_range_inner'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(
        ds[result['qc_variable_name']].attrs['fail_lower_range_inner'],
        limit_value1,
    )
    assert 'fail_upper_range_inner' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['fail_upper_range_inner'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(
        ds[result['qc_variable_name']].attrs['fail_upper_range_inner'],
        limit_value2,
    )

    result = ds.qcfilter.add_inside_test(
        var_name, limit_value1, limit_value2, test_assessment='Indeterminate'
    )
    assert 'warn_lower_range_inner' in ds[result['qc_variable_name']].attrs.keys()
    assert 'warn_upper_range_inner' in ds[result['qc_variable_name']].attrs.keys()

    result = ds.qcfilter.add_inside_test(var_name, limit_value1, limit_value2, use_dask=True)
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 1820693
    result = ds.qcfilter.add_inside_test(
        var_name,
        limit_value1,
        limit_value2,
    )
    data = ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
    assert np.sum(data) == 1820693

    # delta test
    test_limit = 0.05
    result = ds.qcfilter.add_delta_test(
        var_name, test_limit, prepend_text='arm', limit_attr_name='warn_delta'
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert np.ma.count_masked(data) == 175
    assert 'warn_delta' in ds[result['qc_variable_name']].attrs.keys()
    assert (
        ds[result['qc_variable_name']].attrs['warn_delta'].dtype
        == ds[result['variable_name']].values.dtype
    )
    assert np.isclose(ds[result['qc_variable_name']].attrs['warn_delta'], test_limit)

    data = ds.qcfilter.get_masked_data(var_name, rm_assessments=['Suspect', 'Bad'])
    assert np.ma.count_masked(data) == 1355

    result = ds.qcfilter.add_delta_test(var_name, test_limit, test_assessment='Bad')
    assert 'fail_delta' in ds[result['qc_variable_name']].attrs.keys()

    comp_ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    with np.testing.assert_raises(ValueError):
        result = ds.qcfilter.add_difference_test(var_name, 'test')

    with np.testing.assert_raises(ValueError):
        result = ds.qcfilter.add_difference_test(
            var_name,
            {comp_ds.attrs['datastream']: comp_ds},
            var_name,
            diff_limit=None,
        )

    assert ds.qcfilter.add_difference_test(var_name, set_test_regardless=False) is None

    result = ds.qcfilter.add_difference_test(
        var_name,
        {comp_ds.attrs['datastream']: comp_ds},
        var_name,
        diff_limit=1,
        prepend_text='arm',
    )
    data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
    assert 'arm' in result['test_meaning']
    assert not (data.mask).all()

    comp_ds.close()
    ds.close()


def test_qctests_dos():
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'

    data = ds[var_name].values
    data[1000:2400] = data[1000]
    data = np.around(data, decimals=5)
    ds[var_name].values = data
    result = ds.qcfilter.add_persistence_test(var_name)
    qc_var_name = result['qc_variable_name']
    test_meaning = (
        'Data failing persistence test. Standard Deviation over a '
        'window of 10 values less than 0.0001.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'] == [test_meaning]

    # There is a precision issue with hardware/VM used in testing that makes the
    # number of tests tripped different than listed value. The isclose() option is to account for that.
    assert np.isclose(np.sum(ds[qc_var_name].values), 1400, atol=10)

    ds.qcfilter.add_persistence_test(var_name, window=10000, prepend_text='DQO')
    test_meaning = (
        'DQO: Data failing persistence test. Standard Deviation over a window of '
        '4320 values less than 0.0001.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_meaning

    ds.close()
    del ds

    # Test the ignore range in persistence test
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    data = ds[var_name].values
    data[1000:1400] = data[1000]
    data[2000:2400] = 14.2
    data = np.around(data, decimals=5)
    ds[var_name].values = data
    result = ds.qcfilter.add_persistence_test(var_name, window=20, min_periods=20, test_limit=0.01)

    assert np.isclose(np.sum(ds[qc_var_name].values), 779, atol=5)

    del ds[qc_var_name]
    result = ds.qcfilter.add_persistence_test(
        var_name,
        window=20,
        min_periods=20,
        test_limit=0.01,
        ignore_range=[14.8, 13.1],
        test_assessment='Suspect',
    )

    assert np.isclose(np.sum(ds[qc_var_name].values), 398, atol=5)
    assert ds[qc_var_name].attrs['flag_assessments'] == ['Suspect']
    test_meaning = (
        'Data failing persistence test. Standard Deviation over a window '
        'of 20 values less than 0.01.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'] == [test_meaning]

    ds.close()
    del ds


def test_add_atmospheric_pressure_test():
    ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=True)
    ds.load()

    variable = 'atmos_pressure'
    qc_variable = 'qc_' + variable

    data = ds[variable].values
    data[200:250] = data[200:250] + 5
    data[500:550] = data[500:550] - 4.6
    ds[variable].values = data
    result = ds.qcfilter.add_atmospheric_pressure_test(variable)
    assert isinstance(result, dict)
    assert np.sum(ds[qc_variable].values) == 1600

    del ds[qc_variable]
    ds.qcfilter.add_atmospheric_pressure_test(variable, use_dask=True)
    assert np.sum(ds[qc_variable].values) == 100

    ds.close
    del ds


def test_add_step_change_test():
    variable = 'temp_mean'
    qc_variable = f"qc_{variable}"
    ds = read_arm_netcdf(EXAMPLE_MET1, keep_variables=['temp_mean', 'atmos_pressure'])
    ds.load()

    result = ds.qcfilter.add_step_change_test(variable)
    assert result == {
        'test_number': 1,
        'test_meaning': 'Shift in data detected with CUSUM algorithm: k=1.0',
        'test_assessment': 'Indeterminate',
        'qc_variable_name': qc_variable,
        'variable_name': variable,
    }
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=1)
    assert len(np.where(index)[0]) == 0
    assert ds[qc_variable].attrs['flag_meanings'] == [
        'Shift in data detected with CUSUM algorithm: k=1.0'
    ]
    assert ds[qc_variable].attrs['flag_assessments'] == ['Indeterminate']

    data = ds[variable].values
    data[100:] -= 5
    data[600:] += 4
    data[800:] += 10
    data[1000:] -= 2
    ds[variable].values = data

    ds.qcfilter.add_step_change_test(variable)
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=2)
    assert np.all(np.where(index)[0] == [99, 100, 599, 600, 799, 800, 999, 1000])
    assert (
        ds[qc_variable].attrs['flag_meanings'][1]
        == 'Shift in data detected with CUSUM algorithm: k=1.0'
    )
    assert ds[qc_variable].attrs['flag_assessments'][1] == 'Indeterminate'

    ds.qcfilter.add_step_change_test(variable, k=4, prepend_text='ARM')
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=3)
    assert np.all(np.where(index)[0] == [99, 100, 599, 600, 799, 800])
    assert (
        ds[qc_variable].attrs['flag_meanings'][2]
        == 'ARM: Shift in data detected with CUSUM algorithm: k=4'
    )

    ds.qcfilter.add_step_change_test(variable, n_flagged=3)
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=4)
    assert np.all(
        np.where(index)[0] == [99, 100, 101, 599, 600, 601, 799, 800, 801, 999, 1000, 1001]
    )

    ds.qcfilter.add_step_change_test(variable, n_flagged=-1, k=5.1, test_assessment='Suspect')
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=5)
    assert np.all(np.where(index)[0] == np.arange(799, 1440))
    assert (
        ds[qc_variable].attrs['flag_meanings'][4]
        == 'Shift in data detected with CUSUM algorithm: k=5.1'
    )
    assert ds[qc_variable].attrs['flag_assessments'][4] == 'Suspect'

    variable = 'atmos_pressure'
    ds.qcfilter.add_step_change_test(variable, detrend=False)
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=1)
    assert len(np.where(index)[0]) == 0

    ds.close
    del ds

    # Test add_nan keyword
    variable = 'temp_mean'
    ds = read_arm_netcdf(EXAMPLE_MET1, keep_variables=variable)
    data = ds[variable].values
    data[600:] += 2
    ds[variable].values = data

    ds = ds.where((ds["time.hour"] < 3) | (ds["time.hour"] > 5), drop=True)

    ds.qcfilter.add_step_change_test(variable, add_nan=False)
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=1)
    assert np.all(np.where(index)[0] == [179, 180, 419, 420])

    ds.qcfilter.add_step_change_test(variable, add_nan=True)
    index = ds.qcfilter.get_qc_test_mask(var_name=variable, test_number=2)
    assert np.all(np.where(index)[0] == [419, 420])

    del ds
