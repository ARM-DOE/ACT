import copy
from datetime import datetime

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from act.io.arm import read_arm_netcdf
from act.qc.arm import add_dqr_to_qc
from act.qc.qcfilter import parse_bit, set_bit, unset_bit
from act.tests import EXAMPLE_MET1, EXAMPLE_METE40, EXAMPLE_IRT25m20s

try:
    import scikit_posthocs  # noqa

    SCIKIT_POSTHOCS_AVAILABLE = True
except ImportError:
    SCIKIT_POSTHOCS_AVAILABLE = False


def test_qc_test_errors():
    ds = read_arm_netcdf(EXAMPLE_MET1)
    var_name = 'temp_mean'

    assert ds.qcfilter.add_less_test(var_name, None) is None
    assert ds.qcfilter.add_greater_test(var_name, None) is None
    assert ds.qcfilter.add_less_equal_test(var_name, None) is None
    assert ds.qcfilter.add_equal_to_test(var_name, None) is None
    assert ds.qcfilter.add_not_equal_to_test(var_name, None) is None


def test_arm_qc():
    # Test DQR Webservice using known DQR
    variable = 'wspd_vec_mean'
    ds = read_arm_netcdf(EXAMPLE_METE40)
    ds_org = copy.deepcopy(ds)
    qc_variable = ds.qcfilter.check_for_ancillary_qc(variable)

    # DQR webservice does go down, so ensure it properly runs first before testing
    try:
        ds = add_dqr_to_qc(ds)

    except ValueError:
        return

    assert 'Suspect' not in ds[qc_variable].attrs['flag_assessments']
    assert 'Incorrect' not in ds[qc_variable].attrs['flag_assessments']
    assert 'Bad' in ds[qc_variable].attrs['flag_assessments']
    assert 'Indeterminate' in ds[qc_variable].attrs['flag_assessments']

    # Check that defualt will update all variables in DQR
    for var_name in ['wdir_vec_mean', 'wdir_vec_std', 'wspd_arith_mean', 'wspd_vec_mean']:
        qc_var = ds.qcfilter.check_for_ancillary_qc(var_name)
        assert ds[qc_var].attrs['flag_meanings'][-1].startswith('D190529.4')

    # Check that variable keyword works as expected.
    ds = copy.deepcopy(ds_org)
    add_dqr_to_qc(ds, variable=variable)
    qc_var = ds.qcfilter.check_for_ancillary_qc(variable)
    assert ds[qc_var].attrs['flag_meanings'][-1].startswith('D190529.4')
    qc_var = ds.qcfilter.check_for_ancillary_qc('wdir_vec_std')
    assert len(ds[qc_var].attrs['flag_masks']) == 0

    # Check that include and exclude keywords work as expected
    ds = copy.deepcopy(ds_org)
    add_dqr_to_qc(ds, variable=variable, exclude=['D190529.4'])
    assert len(ds[qc_variable].attrs['flag_meanings']) == 4
    add_dqr_to_qc(ds, variable=variable, include=['D400101.1'])
    assert len(ds[qc_variable].attrs['flag_meanings']) == 4
    add_dqr_to_qc(ds, variable=variable, include=['D190529.4'])
    assert len(ds[qc_variable].attrs['flag_meanings']) == 5
    add_dqr_to_qc(ds, variable=variable, assessment='Incorrect')
    assert len(ds[qc_variable].attrs['flag_meanings']) == 5

    # Test additional keywords
    add_dqr_to_qc(
        ds,
        variable=variable,
        assessment='Suspect',
        cleanup_qc=False,
        dqr_link=True,
        skip_location_vars=True,
    )
    assert len(ds[qc_variable].attrs['flag_meanings']) == 6

    # Default is to normalize assessment terms. Check that we can turn off.
    add_dqr_to_qc(ds, variable=variable, normalize_assessment=False)
    assert 'Suspect' in ds[qc_variable].attrs['flag_assessments']

    # Test that an error is raised when no datastream global attributes
    with np.testing.assert_raises(ValueError):
        ds4 = copy.deepcopy(ds)
        del ds4.attrs['datastream']
        del ds4.attrs['_datastream']
        add_dqr_to_qc(ds4, variable=variable)


def test_qcfilter():
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'
    expected_qc_var_name = 'qc_' + var_name

    ds.qcfilter.check_for_ancillary_qc(
        var_name, add_if_missing=True, cleanup=False, flag_type=False
    )
    assert expected_qc_var_name in list(ds.keys())
    del ds[expected_qc_var_name]

    # Perform adding of quality control variables to Xarray dataset
    result = ds.qcfilter.add_test(var_name, test_meaning='Birds!')
    assert isinstance(result, dict)
    qc_var_name = result['qc_variable_name']
    assert qc_var_name == expected_qc_var_name

    # Check that new linking and describing attributes are set
    assert ds[qc_var_name].attrs['standard_name'] == 'quality_flag'
    assert ds[var_name].attrs['ancillary_variables'] == qc_var_name

    # Check that CF attributes are set including new flag_assessments
    assert 'flag_masks' in ds[qc_var_name].attrs.keys()
    assert 'flag_meanings' in ds[qc_var_name].attrs.keys()
    assert 'flag_assessments' in ds[qc_var_name].attrs.keys()

    # Check that the values of the attributes are set correctly
    assert ds[qc_var_name].attrs['flag_assessments'][0] == 'Bad'
    assert ds[qc_var_name].attrs['flag_meanings'][0] == 'Birds!'
    assert ds[qc_var_name].attrs['flag_masks'][0] == 1

    # Set some test values
    index = [0, 1, 2, 30]
    ds.qcfilter.set_test(var_name, index=index, test_number=result['test_number'])

    # Add a new test and set values
    index2 = [6, 7, 8, 50]
    ds.qcfilter.add_test(
        var_name,
        index=index2,
        test_number=9,
        test_meaning='testing high number',
        test_assessment='Suspect',
    )

    # Retrieve data from Xarray dataset as numpy masked array. Count number of masked
    # elements and ensure equal to size of index array.
    data = ds.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
    assert np.ma.count_masked(data) == len(index)

    data = ds.qcfilter.get_masked_data(var_name, rm_assessments='Suspect', return_nan_array=True)
    assert np.sum(np.isnan(data)) == len(index2)

    data = ds.qcfilter.get_masked_data(
        var_name, rm_assessments=['Bad', 'Suspect'], ma_fill_value=np.nan
    )
    assert np.ma.count_masked(data) == len(index + index2)

    # Test internal function for returning the index array of where the
    # tests are set.
    assert (
        np.sum(
            ds.qcfilter.get_qc_test_mask(var_name, result['test_number'], return_index=True)
            - np.array(index, dtype=int)
        )
        == 0
    )

    # Test adding QC for length-1 variables
    ds['west'] = ('west', ['W'])
    ds['avg_wind_speed'] = ('west', [20])

    # Should not fail the test
    ds.qcfilter.add_test(
        'avg_wind_speed',
        index=ds.avg_wind_speed.data > 100,
        test_meaning='testing bool flag: false',
        test_assessment='Suspect',
    )
    assert ds.qc_avg_wind_speed.data == 0

    # Should fail the test
    ds.qcfilter.add_test(
        'avg_wind_speed',
        index=ds.avg_wind_speed.data < 100,
        test_meaning='testing bool flag: true',
        test_assessment='Suspect',
    )
    assert ds.qc_avg_wind_speed.data == 2

    # Should fail the test
    ds.qcfilter.add_test(
        'avg_wind_speed',
        index=[0],
        test_meaning='testing idx flag: true',
        test_assessment='Suspect',
    )
    assert ds.qc_avg_wind_speed.data == 6

    # Should not fail the test
    ds.qcfilter.add_test(
        'avg_wind_speed',
        test_meaning='testing idx flag: false',
        test_assessment='Suspect',
    )
    assert ds.qc_avg_wind_speed.data == 6

    # Unset a test
    ds.qcfilter.unset_test(var_name, index=0, test_number=result['test_number'])
    # Remove the test
    ds.qcfilter.remove_test(var_name, test_number=33)

    # Ensure removal works when flag_masks is a numpy array
    ds['qc_' + var_name].attrs['flag_masks'] = np.array(ds['qc_' + var_name].attrs['flag_masks'])
    ds.qcfilter.remove_test(var_name, test_number=result['test_number'])
    pytest.raises(ValueError, ds.qcfilter.add_test, var_name)
    pytest.raises(ValueError, ds.qcfilter.remove_test, var_name)

    ds.close()

    assert np.all(parse_bit([257]) == np.array([1, 9], dtype=np.int32))
    pytest.raises(ValueError, parse_bit, [1, 2])
    pytest.raises(ValueError, parse_bit, -1)

    assert set_bit(0, 16) == 32768
    data = range(0, 4)
    assert isinstance(set_bit(list(data), 2), list)
    assert isinstance(set_bit(tuple(data), 2), tuple)
    assert isinstance(unset_bit(list(data), 2), list)
    assert isinstance(unset_bit(tuple(data), 2), tuple)

    # Fill in missing tests
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    del ds[var_name].attrs['long_name']
    # Test creating a qc variable
    ds.qcfilter.create_qc_variable(var_name)
    # Test creating a second qc variable and of flag type
    ds.qcfilter.create_qc_variable(var_name, flag_type=True)
    result = ds.qcfilter.add_test(
        var_name,
        index=[1, 2, 3],
        test_number=9,
        test_meaning='testing high number',
        flag_value=True,
    )
    ds.qcfilter.set_test(var_name, index=5, test_number=9, flag_value=True)
    data = ds.qcfilter.get_masked_data(var_name)
    assert np.isclose(np.sum(data), 42674.766, 0.01)
    data = ds.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
    assert np.isclose(np.sum(data), 42643.195, 0.01)

    ds.qcfilter.unset_test(var_name, test_number=9, flag_value=True)
    ds.qcfilter.unset_test(var_name, index=1, test_number=9, flag_value=True)
    assert ds.qcfilter.available_bit(result['qc_variable_name']) == 10
    assert ds.qcfilter.available_bit(result['qc_variable_name'], recycle=True) == 1
    ds.qcfilter.remove_test(var_name, test_number=9, flag_value=True)

    ds.qcfilter.update_ancillary_variable(var_name)
    # Test updating ancillary variable if does not exist
    ds.qcfilter.update_ancillary_variable('not_a_variable_name')
    # Change ancillary_variables attribute to test if add correct qc variable correctly
    ds[var_name].attrs['ancillary_variables'] = 'a_different_name'
    ds.qcfilter.update_ancillary_variable(var_name, qc_var_name=expected_qc_var_name)
    assert expected_qc_var_name in ds[var_name].attrs['ancillary_variables']

    # Test flag QC
    var_name = 'inst_sfc_ir_temp'
    qc_var_name = 'qc_' + var_name
    ds.qcfilter.create_qc_variable(var_name, flag_type=True)
    assert qc_var_name in list(ds.data_vars)
    assert 'flag_values' in ds[qc_var_name].attrs.keys()
    assert 'flag_masks' not in ds[qc_var_name].attrs.keys()
    del ds[qc_var_name]

    qc_var_name = ds.qcfilter.check_for_ancillary_qc(
        var_name, add_if_missing=True, cleanup=False, flag_type=True
    )
    assert qc_var_name in list(ds.data_vars)
    assert 'flag_values' in ds[qc_var_name].attrs.keys()
    assert 'flag_masks' not in ds[qc_var_name].attrs.keys()
    del ds[qc_var_name]

    ds.qcfilter.add_missing_value_test(var_name, flag_value=True, prepend_text='arm')
    ds.qcfilter.add_test(
        var_name,
        index=list(range(0, 20)),
        test_number=2,
        test_meaning='Testing flag',
        flag_value=True,
        test_assessment='Suspect',
    )
    assert qc_var_name in list(ds.data_vars)
    assert 'flag_values' in ds[qc_var_name].attrs.keys()
    assert 'flag_masks' not in ds[qc_var_name].attrs.keys()
    assert 'standard_name' in ds[qc_var_name].attrs.keys()
    assert ds[qc_var_name].attrs['flag_values'] == [1, 2]
    assert ds[qc_var_name].attrs['flag_assessments'] == ['Bad', 'Suspect']

    ds.close()


@pytest.mark.skipif(not SCIKIT_POSTHOCS_AVAILABLE, reason='scikit_posthocs is not installed.')
def test_qcfilter2():
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'
    expected_qc_var_name = 'qc_' + var_name

    data = ds[var_name].values
    data[0:4] = data[0:4] + 30.0
    data[1000:1024] = data[1000:1024] + 30.0
    ds[var_name].values = data

    coef = 1.4
    ds.qcfilter.add_iqr_test(var_name, coef=1.4, test_assessment='Bad', prepend_text='arm')
    assert np.sum(ds[expected_qc_var_name].values) == 28
    assert ds[expected_qc_var_name].attrs['flag_masks'] == [1]
    assert ds[expected_qc_var_name].attrs['flag_meanings'] == [
        f'arm: Value outside of interquartile range test range with a coefficient of {coef}'
    ]

    ds.qcfilter.add_iqr_test(var_name, test_number=3, prepend_text='ACT')
    assert np.sum(ds[expected_qc_var_name].values) == 140
    assert ds[expected_qc_var_name].attrs['flag_masks'] == [1, 4]
    assert ds[expected_qc_var_name].attrs['flag_meanings'][-1] == (
        'ACT: Value outside of interquartile range test range with a coefficient of 1.5'
    )

    ds.qcfilter.add_gesd_test(var_name, test_assessment='Bad')
    assert np.sum(ds[expected_qc_var_name].values) == 204
    assert ds[expected_qc_var_name].attrs['flag_masks'] == [1, 4, 8]
    assert ds[expected_qc_var_name].attrs['flag_meanings'][-1] == (
        'Value failed generalized Extreme Studentized Deviate test with an alpha of 0.05'
    )

    ds.qcfilter.add_gesd_test(var_name, alpha=0.1)
    assert np.sum(ds[expected_qc_var_name].values) == 332
    assert ds[expected_qc_var_name].attrs['flag_masks'] == [1, 4, 8, 16]
    assert ds[expected_qc_var_name].attrs['flag_meanings'][-1] == (
        'Value failed generalized Extreme Studentized Deviate test with an alpha of 0.1'
    )
    assert ds[expected_qc_var_name].attrs['flag_assessments'] == [
        'Bad',
        'Indeterminate',
        'Bad',
        'Indeterminate',
    ]


def test_qcfilter3():
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'
    result = ds.qcfilter.add_test(var_name, index=range(0, 100), test_meaning='testing')
    qc_var_name = result['qc_variable_name']
    assert ds[qc_var_name].values.dtype.kind in np.typecodes['AllInteger']

    ds[qc_var_name].values = ds[qc_var_name].values.astype(np.float32)
    assert ds[qc_var_name].values.dtype.kind not in np.typecodes['AllInteger']

    result = ds.qcfilter.get_qc_test_mask(var_name=var_name, test_number=1, return_index=False)
    assert np.sum(result) == 100
    result = ds.qcfilter.get_qc_test_mask(var_name=var_name, test_number=1, return_index=True)
    assert np.sum(result) == 4950

    # Test where QC variables are not integer type
    ds = ds.resample(time='5min').mean(keep_attrs=True)
    ds.qcfilter.add_test(var_name, index=range(0, ds.time.size), test_meaning='Testing float')
    assert np.sum(ds[qc_var_name].values) == 582

    ds[qc_var_name].values = ds[qc_var_name].values.astype(np.float32)
    ds.qcfilter.remove_test(var_name, test_number=2)
    assert np.sum(ds[qc_var_name].values) == 6


def test_qc_speed():
    """
    This tests the speed of the QC module to ensure changes do not significantly
    slow down the module's processing.
    """

    n_variables = 100
    n_samples = 100

    time = pd.date_range(start='2022-02-17 00:00:00', end='2022-02-18 00:00:00', periods=n_samples)

    # Create data variables with random noise
    np.random.seed(42)
    noisy_data_mapping = {f'data_var_{i}': np.random.random(time.shape) for i in range(n_variables)}

    ds = xr.Dataset(
        data_vars={name: ('time', data) for name, data in noisy_data_mapping.items()},
        coords={'time': time},
    )

    start = datetime.utcnow()
    for name, var in noisy_data_mapping.items():
        failed_qc = var > 0.75  # Consider data above 0.75 as bad. Negligible time here.
        ds.qcfilter.add_test(name, index=failed_qc, test_meaning='Value above threshold')

    time_diff = datetime.utcnow() - start
    assert time_diff.seconds <= 4


def test_datafilter():
    ds = read_arm_netcdf(EXAMPLE_MET1, drop_variables=['base_time', 'time_offset'])
    ds.clean.cleanup()

    data_var_names = list(ds.data_vars)
    qc_var_names = [var_name for var_name in ds.data_vars if var_name.startswith('qc_')]
    data_var_names = list(set(data_var_names) - set(qc_var_names))
    data_var_names.sort()
    qc_var_names.sort()

    var_name = 'atmos_pressure'

    ds_1 = ds.mean()

    ds.qcfilter.add_less_test(var_name, 99, test_assessment='Bad')
    ds_filtered = copy.deepcopy(ds)
    ds_filtered.qcfilter.datafilter(rm_assessments='Bad')
    ds_2 = ds_filtered.mean()
    assert np.isclose(ds_1[var_name].values, 98.86, atol=0.01)
    assert np.isclose(ds_2[var_name].values, 99.15, atol=0.01)
    assert isinstance(ds_1[var_name].data, da.core.Array)
    assert 'act.qc.datafilter' in ds_filtered[var_name].attrs['history']

    ds_filtered = copy.deepcopy(ds)
    ds_filtered.qcfilter.datafilter(rm_assessments='Bad', variables=var_name, del_qc_var=True)
    ds_2 = ds_filtered.mean()
    assert np.isclose(ds_2[var_name].values, 99.15, atol=0.01)
    expected_var_names = sorted(list(set(data_var_names + qc_var_names) - {'qc_' + var_name}))
    assert sorted(list(ds_filtered.data_vars)) == expected_var_names

    ds_filtered = copy.deepcopy(ds)
    ds_filtered.qcfilter.datafilter(rm_assessments='Bad', del_qc_var=True)
    assert sorted(list(ds_filtered.data_vars)) == data_var_names

    ds.close()
    del ds


def test_qc_data_type():
    drop_vars = [
        'base_time',
        'time_offset',
        'inst_up_long_case_resist',
        'inst_up_long_hemisp_tp',
        'inst_up_short_hemisp_tp',
        'inst_sfc_ir_temp',
        'lat',
        'lon',
        'alt',
    ]
    ds = read_arm_netcdf(EXAMPLE_IRT25m20s, drop_variables=drop_vars)
    var_name = 'inst_up_long_dome_resist'
    expected_qc_var_name = 'qc_' + var_name
    ds.qcfilter.check_for_ancillary_qc(var_name, add_if_missing=True)
    del ds[expected_qc_var_name].attrs['flag_meanings']
    del ds[expected_qc_var_name].attrs['flag_assessments']
    ds[expected_qc_var_name] = ds[expected_qc_var_name].astype(np.int8)
    ds.qcfilter.add_test(var_name, index=[1], test_number=9, test_meaning='First test')

    assert ds[expected_qc_var_name].attrs['flag_masks'][0].dtype == np.uint32
    assert ds[expected_qc_var_name].dtype == np.int16
    ds.qcfilter.add_test(var_name, index=[1], test_number=17, test_meaning='Second test')
    assert ds[expected_qc_var_name].dtype == np.int32
    ds.qcfilter.add_test(var_name, index=[1], test_number=33, test_meaning='Third test')
    assert ds[expected_qc_var_name].dtype == np.int64
    assert ds[expected_qc_var_name].attrs['flag_masks'][0].dtype == np.uint64

    ds.qcfilter.add_test(var_name, index=[1], test_meaning='Fourth test', recycle=True)
