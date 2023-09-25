import copy
from datetime import datetime
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pathlib import Path

from act.io.armfiles import read_netcdf
from act.qc.arm import add_dqr_to_qc
from act.qc.qcfilter import parse_bit, set_bit, unset_bit
from act.qc.radiometer_tests import fft_shading_test
from act.qc.sp2 import SP2ParticleCriteria, PYSP2_AVAILABLE
from act.tests import (
    EXAMPLE_CEIL1,
    EXAMPLE_CO2FLX4M,
    EXAMPLE_MET1,
    EXAMPLE_METE40,
    EXAMPLE_MFRSR,
    EXAMPLE_IRT25m20s,
    EXAMPLE_BRS,
    EXAMPLE_MET_YAML,
    EXAMPLE_ENA_MET
)
from act.qc.bsrn_tests import _calculate_solar_parameters
from act.qc.add_supplemental_qc import read_yaml_supplemental_qc, apply_supplemental_qc

try:
    import scikit_posthocs
    SCIKIT_POSTHOCS_AVAILABLE = True
except ImportError:
    SCIKIT_POSTHOCS_AVAILABLE = False


def test_fft_shading_test():
    ds = read_netcdf(EXAMPLE_MFRSR)
    ds.clean.cleanup()
    ds = fft_shading_test(ds)
    qc_data = ds['qc_diffuse_hemisp_narrowband_filter4']
    assert np.nansum(qc_data.values) == 7164


def test_global_qc_cleanup():
    ds = read_netcdf(EXAMPLE_MET1)
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


def test_qc_test_errors():
    ds = read_netcdf(EXAMPLE_MET1)
    var_name = 'temp_mean'

    assert ds.qcfilter.add_less_test(var_name, None) is None
    assert ds.qcfilter.add_greater_test(var_name, None) is None
    assert ds.qcfilter.add_less_equal_test(var_name, None) is None
    assert ds.qcfilter.add_equal_to_test(var_name, None) is None
    assert ds.qcfilter.add_not_equal_to_test(var_name, None) is None


def test_arm_qc():
    # Test DQR Webservice using known DQR
    variable = 'wspd_vec_mean'
    qc_variable = 'qc_' + variable
    ds = read_netcdf(EXAMPLE_METE40)

    # DQR webservice does go down, so ensure it
    # properly runs first before testing
    try:
        ds = add_dqr_to_qc(ds, variable=variable)
        ran = True
        ds.attrs['_datastream'] = ds.attrs['datastream']
        del ds.attrs['datastream']
        ds2 = add_dqr_to_qc(ds, variable=variable)
        ds3 = add_dqr_to_qc(ds)
        add_dqr_to_qc(ds, variable=variable, exclude=['D190529.4'])
        add_dqr_to_qc(ds, variable=variable, include=['D400101.1'])
        with np.testing.assert_raises(ValueError):
            del ds.attrs['_datastream']
            add_dqr_to_qc(ds, variable=variable)

    except ValueError:
        ran = False

    if ran:
        assert qc_variable in ds
        dqr = [True for d in ds[qc_variable].attrs['flag_meanings'] if 'D190529.4' in d]
        assert dqr[0] is True
        assert 'Suspect' not in ds[qc_variable].attrs['flag_assessments']
        assert 'Incorrect' not in ds[qc_variable].attrs['flag_assessments']

        assert qc_variable in ds2
        dqr = [True for d in ds2[qc_variable].attrs['flag_meanings'] if 'D190529.4' in d]
        assert dqr[0] is True
        assert 'Suspect' not in ds2[qc_variable].attrs['flag_assessments']
        assert 'Incorrect' not in ds2[qc_variable].attrs['flag_assessments']

        assert qc_variable in ds3
        dqr = [True for d in ds3[qc_variable].attrs['flag_meanings'] if 'D190529.4' in d]
        assert dqr[0] is True
        assert 'Suspect' not in ds3[qc_variable].attrs['flag_assessments']
        assert 'Incorrect' not in ds3[qc_variable].attrs['flag_assessments']


def test_qcfilter():
    ds = read_netcdf(EXAMPLE_IRT25m20s)
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

    data = ds.qcfilter.get_masked_data(
        var_name, rm_assessments='Suspect', return_nan_array=True
    )
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
    ds = read_netcdf(EXAMPLE_IRT25m20s)
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


@pytest.mark.skipif(not SCIKIT_POSTHOCS_AVAILABLE,
                    reason="scikit_posthocs is not installed.")
def test_qcfilter2():
    ds = read_netcdf(EXAMPLE_IRT25m20s)
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
    ds = read_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'
    result = ds.qcfilter.add_test(var_name, index=range(0, 100), test_meaning='testing')
    qc_var_name = result['qc_variable_name']
    assert ds[qc_var_name].values.dtype.kind in np.typecodes['AllInteger']

    ds[qc_var_name].values = ds[qc_var_name].values.astype(np.float32)
    assert ds[qc_var_name].values.dtype.kind not in np.typecodes['AllInteger']

    result = ds.qcfilter.get_qc_test_mask(
        var_name=var_name, test_number=1, return_index=False
    )
    assert np.sum(result) == 100
    result = ds.qcfilter.get_qc_test_mask(
        var_name=var_name, test_number=1, return_index=True
    )
    assert np.sum(result) == 4950

    # Test where QC variables are not integer type
    ds = ds.resample(time='5min').mean(keep_attrs=True)
    ds.qcfilter.add_test(
        var_name, index=range(0, ds.time.size), test_meaning='Testing float'
    )
    assert np.sum(ds[qc_var_name].values) == 582

    ds[qc_var_name].values = ds[qc_var_name].values.astype(np.float32)
    ds.qcfilter.remove_test(var_name, test_number=2)
    assert np.sum(ds[qc_var_name].values) == 6


def test_qctests():
    ds = read_netcdf(EXAMPLE_IRT25m20s)
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

    result = ds.qcfilter.add_equal_to_test(
        var_name, limit_value, test_assessment='Indeterminate'
    )
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

    result = ds.qcfilter.add_outside_test(
        var_name, limit_value1, limit_value2, use_dask=True
    )
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

    comp_ds = read_netcdf(EXAMPLE_IRT25m20s)
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
    ds = read_netcdf(EXAMPLE_IRT25m20s)
    var_name = 'inst_up_long_dome_resist'

    # persistence test
    data = ds[var_name].values
    data[1000: 2400] = data[1000]
    data = np.around(data, decimals=3)
    ds[var_name].values = data
    result = ds.qcfilter.add_persistence_test(var_name)
    qc_var_name = result['qc_variable_name']
    test_meaning = (
        'Data failing persistence test. Standard Deviation over a '
        'window of 10 values less than 0.0001.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_meaning
    # There is a precision issue with GitHub testing that makes the number of tests
    # tripped off. This isclose() option is to account for that.
    assert np.isclose(np.sum(ds[qc_var_name].values), 1399, atol=2)

    ds.qcfilter.add_persistence_test(var_name, window=10000, prepend_text='DQO')
    test_meaning = (
        'DQO: Data failing persistence test. Standard Deviation over a window of '
        '4320 values less than 0.0001.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_meaning


def test_datafilter():
    ds = read_netcdf(EXAMPLE_MET1, drop_variables=['base_time', 'time_offset'])
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
    ds_filtered.qcfilter.datafilter(rm_assessments='Bad', del_qc_var=False)
    ds_2 = ds_filtered.mean()
    assert np.isclose(ds_1[var_name].values, 98.86, atol=0.01)
    assert np.isclose(ds_2[var_name].values, 99.15, atol=0.01)
    assert isinstance(ds_1[var_name].data, da.core.Array)
    assert 'act.qc.datafilter' in ds_filtered[var_name].attrs['history']

    ds_filtered = copy.deepcopy(ds)
    ds_filtered.qcfilter.datafilter(rm_assessments='Bad', variables=var_name)
    ds_2 = ds_filtered.mean()
    assert np.isclose(ds_2[var_name].values, 99.15, atol=0.01)
    expected_var_names = sorted(list(set(data_var_names + qc_var_names) - set(['qc_' + var_name])))
    assert sorted(list(ds_filtered.data_vars)) == expected_var_names

    ds_filtered = copy.deepcopy(ds)
    ds_filtered.qcfilter.datafilter(rm_assessments='Bad', del_qc_var=True)
    assert sorted(list(ds_filtered.data_vars)) == data_var_names

    ds.close()
    del ds


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


def test_clean():
    # Read test data
    ceil_ds = read_netcdf([EXAMPLE_CEIL1])
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


def test_compare_time_series_trends():

    drop_vars = [
        'base_time',
        'time_offset',
        'atmos_pressure',
        'qc_atmos_pressure',
        'temp_std',
        'rh_mean',
        'qc_rh_mean',
        'rh_std',
        'vapor_pressure_mean',
        'qc_vapor_pressure_mean',
        'vapor_pressure_std',
        'wspd_arith_mean',
        'qc_wspd_arith_mean',
        'wspd_vec_mean',
        'qc_wspd_vec_mean',
        'wdir_vec_mean',
        'qc_wdir_vec_mean',
        'wdir_vec_std',
        'tbrg_precip_total',
        'qc_tbrg_precip_total',
        'tbrg_precip_total_corr',
        'qc_tbrg_precip_total_corr',
        'org_precip_rate_mean',
        'qc_org_precip_rate_mean',
        'pwd_err_code',
        'pwd_mean_vis_1min',
        'qc_pwd_mean_vis_1min',
        'pwd_mean_vis_10min',
        'qc_pwd_mean_vis_10min',
        'pwd_pw_code_inst',
        'qc_pwd_pw_code_inst',
        'pwd_pw_code_15min',
        'qc_pwd_pw_code_15min',
        'pwd_pw_code_1hr',
        'qc_pwd_pw_code_1hr',
        'pwd_precip_rate_mean_1min',
        'qc_pwd_precip_rate_mean_1min',
        'pwd_cumul_rain',
        'qc_pwd_cumul_rain',
        'pwd_cumul_snow',
        'qc_pwd_cumul_snow',
        'logger_volt',
        'qc_logger_volt',
        'logger_temp',
        'qc_logger_temp',
        'lat',
        'lon',
        'alt',
    ]
    ds = read_netcdf(EXAMPLE_MET1, drop_variables=drop_vars)
    ds.clean.cleanup()
    ds2 = copy.deepcopy(ds)

    var_name = 'temp_mean'
    qc_var_name = ds.qcfilter.check_for_ancillary_qc(
        var_name, add_if_missing=False, cleanup=False, flag_type=False
    )
    ds.qcfilter.compare_time_series_trends(
        var_name=var_name,
        time_shift=60,
        comp_var_name=var_name,
        comp_dataset=ds2,
        time_qc_threshold=60 * 10,
    )

    test_description = (
        'Time shift detected with Minimum Difference test. Comparison of '
        'temp_mean with temp_mean off by 0 seconds exceeding absolute '
        'threshold of 600 seconds.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_description

    time = ds2['time'].values + np.timedelta64(1, 'h')
    time_attrs = ds2['time'].attrs
    ds2 = ds2.assign_coords({'time': time})
    ds2['time'].attrs = time_attrs

    ds.qcfilter.compare_time_series_trends(
        var_name=var_name, comp_dataset=ds2, time_step=60, time_match_threshhold=50
    )

    test_description = (
        'Time shift detected with Minimum Difference test. Comparison of '
        'temp_mean with temp_mean off by 3600 seconds exceeding absolute '
        'threshold of 900 seconds.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_description


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
    ds = read_netcdf(EXAMPLE_IRT25m20s, drop_variables=drop_vars)
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


@pytest.mark.skipif(not PYSP2_AVAILABLE, reason="PySP2 is not installed.")
def test_sp2_particle_config():
    particle_config_ds = SP2ParticleCriteria()
    assert particle_config_ds.ScatMaxPeakHt1 == 60000
    assert particle_config_ds.ScatMinPeakHt1 == 250
    assert particle_config_ds.ScatMaxPeakHt2 == 60000
    assert particle_config_ds.ScatMinPeakHt2 == 250
    assert particle_config_ds.ScatMinWidth == 10
    assert particle_config_ds.ScatMaxWidth == 90
    assert particle_config_ds.ScatMinPeakPos == 20
    assert particle_config_ds.ScatMaxPeakPos == 90
    assert particle_config_ds.IncanMinPeakHt1 == 200
    assert particle_config_ds.IncanMinPeakHt2 == 200
    assert particle_config_ds.IncanMaxPeakHt1 == 60000
    assert particle_config_ds.IncanMaxPeakHt2 == 60000
    assert particle_config_ds.IncanMinWidth == 5
    assert particle_config_ds.IncanMaxWidth == np.inf
    assert particle_config_ds.IncanMinPeakPos == 20
    assert particle_config_ds.IncanMaxPeakPos == 90
    assert particle_config_ds.IncanMinPeakRatio == 0.1
    assert particle_config_ds.IncanMaxPeakRatio == 25
    assert particle_config_ds.IncanMaxPeakOffset == 11
    assert particle_config_ds.c0Mass1 == 0
    assert particle_config_ds.c1Mass1 == 0.0001896
    assert particle_config_ds.c2Mass1 == 0
    assert particle_config_ds.c3Mass1 == 0
    assert particle_config_ds.c0Mass2 == 0
    assert particle_config_ds.c1Mass2 == 0.0016815
    assert particle_config_ds.c2Mass2 == 0
    assert particle_config_ds.c3Mass2 == 0
    assert particle_config_ds.c0Scat1 == 0
    assert particle_config_ds.c1Scat1 == 78.141
    assert particle_config_ds.c2Scat1 == 0
    assert particle_config_ds.c0Scat2 == 0
    assert particle_config_ds.c1Scat2 == 752.53
    assert particle_config_ds.c2Scat2 == 0
    assert particle_config_ds.densitySO4 == 1.8
    assert particle_config_ds.densityBC == 1.8
    assert particle_config_ds.TempSTP == 273.15
    assert particle_config_ds.PressSTP == 1013.25


def test_bsrn_limits_test():

    for use_dask in [False, True]:
        ds = read_netcdf(EXAMPLE_BRS)
        var_names = list(ds.data_vars)
        # Remove QC variables to make testing easier
        for var_name in var_names:
            if var_name.startswith('qc_'):
                del ds[var_name]

        # Add atmospheric temperature fake data
        ds['temp_mean'] = xr.DataArray(
            data=np.full(ds.time.size, 13.5), dims=['time'],
            attrs={'long_name': 'Atmospheric air temperature', 'units': 'degC'})

        # Make a short direct variable since BRS does not have one
        ds['short_direct'] = copy.deepcopy(ds['short_direct_normal'])
        ds['short_direct'].attrs['ancillary_variables'] = 'qc_short_direct'
        ds['short_direct'].attrs['long_name'] = 'Shortwave direct irradiance, pyrheliometer'
        sza, Sa = _calculate_solar_parameters(ds, 'lat', 'lon', 1360.8)
        ds['short_direct'].data = ds['short_direct'].data * .5

        # Make up long variable since BRS does not have values
        ds['up_long_hemisp'].data = copy.deepcopy(ds['down_long_hemisp_shaded'].data)
        data = copy.deepcopy(ds['down_short_hemisp'].data)
        ds['up_short_hemisp'].data = data

        # Test that nothing happens when no variable names are provided
        ds.qcfilter.bsrn_limits_test()

        # Mess with data to get tests to trip
        data = ds['down_short_hemisp'].values
        data[200:300] -= 10
        data[800:850] += 330
        data[1340:1380] += 600
        ds['down_short_hemisp'].data = da.from_array(data)

        data = ds['down_short_diffuse_hemisp'].values
        data[200:250] = data[200:250] - 1.9
        data[250:300] = data[250:300] - 3.9
        data[800:850] += 330
        data[1340:1380] += 600
        ds['down_short_diffuse_hemisp'].data = da.from_array(data)

        data = ds['short_direct_normal'].values
        data[200:250] = data[200:250] - 1.9
        data[250:300] = data[250:300] - 3.9
        data[800:850] += 600
        data[1340:1380] += 800
        ds['short_direct_normal'].data = da.from_array(data)

        data = ds['short_direct'].values
        data[200:250] = data[200:250] - 1.9
        data[250:300] = data[250:300] - 3.9
        data[800:850] += 300
        data[1340:1380] += 800
        ds['short_direct'].data = da.from_array(data)

        data = ds['down_long_hemisp_shaded'].values
        data[200:250] = data[200:250] - 355
        data[250:300] = data[250:300] - 400
        data[800:850] += 200
        data[1340:1380] += 400
        ds['down_long_hemisp_shaded'].data = da.from_array(data)

        data = ds['up_long_hemisp'].values
        data[200:250] = data[200:250] - 355
        data[250:300] = data[250:300] - 400
        data[800:850] += 300
        data[1340:1380] += 500
        ds['up_long_hemisp'].data = da.from_array(data)

        ds.qcfilter.bsrn_limits_test(
            gbl_SW_dn_name='down_short_hemisp',
            glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
            direct_normal_SW_dn_name='short_direct_normal',
            glb_SW_up_name='up_short_hemisp',
            glb_LW_dn_name='down_long_hemisp_shaded',
            glb_LW_up_name='up_long_hemisp',
            direct_SW_dn_name='short_direct',
            use_dask=use_dask)

        assert ds['qc_down_short_hemisp'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_down_short_hemisp'].attrs['flag_meanings'][-2] == \
            'Value less than BSRN physically possible limit of -4.0 W/m^2'
        assert ds['qc_down_short_hemisp'].attrs['flag_meanings'][-1] == \
            'Value greater than BSRN physically possible limit'

        assert ds['qc_down_short_diffuse_hemisp'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_down_short_diffuse_hemisp'].attrs['flag_assessments'] == ['Bad', 'Bad']

        assert ds['qc_short_direct'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_short_direct'].attrs['flag_assessments'] == ['Bad', 'Bad']
        assert ds['qc_short_direct'].attrs['flag_meanings'] == \
            ['Value less than BSRN physically possible limit of -4.0 W/m^2',
             'Value greater than BSRN physically possible limit']

        assert ds['qc_short_direct_normal'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_short_direct_normal'].attrs['flag_meanings'][-1] == \
            'Value greater than BSRN physically possible limit'

        assert ds['qc_down_short_hemisp'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_down_short_hemisp'].attrs['flag_meanings'][-1] == \
            'Value greater than BSRN physically possible limit'

        assert ds['qc_up_short_hemisp'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_up_short_hemisp'].attrs['flag_meanings'][-1] == \
            'Value greater than BSRN physically possible limit'

        assert ds['qc_up_long_hemisp'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_up_long_hemisp'].attrs['flag_meanings'][-1] == \
            'Value greater than BSRN physically possible limit of 900.0 W/m^2'

        ds.qcfilter.bsrn_limits_test(
            test="Extremely Rare",
            gbl_SW_dn_name='down_short_hemisp',
            glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
            direct_normal_SW_dn_name='short_direct_normal',
            glb_SW_up_name='up_short_hemisp',
            glb_LW_dn_name='down_long_hemisp_shaded',
            glb_LW_up_name='up_long_hemisp',
            direct_SW_dn_name='short_direct',
            use_dask=use_dask)

        assert ds['qc_down_short_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_down_short_diffuse_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_short_direct'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_short_direct_normal'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_up_short_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_up_long_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]

        assert ds['qc_up_long_hemisp'].attrs['flag_meanings'][-1] == \
            'Value greater than BSRN extremely rare limit of 700.0 W/m^2'

        assert ds['qc_down_long_hemisp_shaded'].attrs['flag_meanings'][-1] == \
            'Value greater than BSRN extremely rare limit of 500.0 W/m^2'

        # down_short_hemisp
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=1)
        assert np.sum(result) == 100
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=2)
        assert np.sum(result) == 26
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=3)
        assert np.sum(result) == 337
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=4)
        assert np.sum(result) == 66

        # down_short_diffuse_hemisp
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=1)
        assert np.sum(result) == 50
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=2)
        assert np.sum(result) == 56
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=3)
        assert np.sum(result) == 100
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=4)
        assert np.sum(result) == 90

        # short_direct_normal
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=1)
        assert np.sum(result) == 46
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=2)
        assert np.sum(result) == 26
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=3)
        assert np.sum(result) == 94
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=4)
        assert np.sum(result) == 38

        # short_direct_normal
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=1)
        assert np.sum(result) == 41
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=2)
        assert np.sum(result) == 607
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=3)
        assert np.sum(result) == 89
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=4)
        assert np.sum(result) == 79

        # down_long_hemisp_shaded
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=1)
        assert np.sum(result) == 50
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=2)
        assert np.sum(result) == 40
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=3)
        assert np.sum(result) == 89
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=4)
        assert np.sum(result) == 90

        # up_long_hemisp
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=1)
        assert np.sum(result) == 50
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=2)
        assert np.sum(result) == 40
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=3)
        assert np.sum(result) == 89
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=4)
        assert np.sum(result) == 90

        # Change data values to trip tests
        ds['down_short_diffuse_hemisp'].values[0:100] = \
            ds['down_short_diffuse_hemisp'].values[0:100] + 100
        ds['up_long_hemisp'].values[0:100] = \
            ds['up_long_hemisp'].values[0:100] - 200

        ds.qcfilter.bsrn_comparison_tests(
            ['Global over Sum SW Ratio', 'Diffuse Ratio', 'SW up', 'LW down to air temp',
             'LW up to air temp', 'LW down to LW up'],
            gbl_SW_dn_name='down_short_hemisp',
            glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
            direct_normal_SW_dn_name='short_direct_normal',
            glb_SW_up_name='up_short_hemisp',
            glb_LW_dn_name='down_long_hemisp_shaded',
            glb_LW_up_name='up_long_hemisp',
            air_temp_name='temp_mean',
            test_assessment='Indeterminate',
            lat_name='lat',
            lon_name='lon',
            use_dask=use_dask
        )

        # Ratio of Global over Sum SW
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=5)
        assert np.sum(result) == 190

        # Diffuse Ratio
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=6)
        assert np.sum(result) == 47

        # Shortwave up comparison
        result = ds.qcfilter.get_qc_test_mask('up_short_hemisp', test_number=5)
        assert np.sum(result) == 226

        # Longwave up to air temperature comparison
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=5)
        assert np.sum(result) == 290

        # Longwave down to air temperature compaison
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=5)
        assert np.sum(result) == 976

        # Lonwave down to longwave up comparison
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=6)
        assert np.sum(result) == 100


def test_add_atmospheric_pressure_test():
    ds = read_netcdf(EXAMPLE_MET1, cleanup_qc=True)
    ds.load()

    variable = 'atmos_pressure'
    qc_varialbe = 'qc_' + variable

    data = ds[variable].values
    data[200:250] = data[200:250] + 5
    data[500:550] = data[500:550] - 4.6
    ds[variable].values = data
    result = ds.qcfilter.add_atmospheric_pressure_test(variable)
    assert isinstance(result, dict)
    assert np.sum(ds[qc_varialbe].values) == 1600

    del ds[qc_varialbe]
    ds.qcfilter.add_atmospheric_pressure_test(variable, use_dask=True)
    assert np.sum(ds[qc_varialbe].values) == 100

    ds.close
    del ds


def test_read_yaml_supplemental_qc():
    ds = read_netcdf(EXAMPLE_MET1, keep_variables=['temp_mean', 'qc_temp_mean'], cleanup_qc=True)

    result = read_yaml_supplemental_qc(ds, EXAMPLE_MET_YAML)
    assert isinstance(result, dict)
    assert len(result.keys()) == 3

    result = read_yaml_supplemental_qc(ds, Path(EXAMPLE_MET_YAML).parent, variables='temp_mean',
                                       assessments=['Bad', 'Incorrect', 'Suspect'])
    assert len(result.keys()) == 2
    assert sorted(result['temp_mean'].keys()) == ['Bad', 'Suspect']

    result = read_yaml_supplemental_qc(ds, 'sgpmetE13.b1.yaml', quiet=True)
    assert result is None

    apply_supplemental_qc(ds, EXAMPLE_MET_YAML)
    assert ds['qc_temp_mean'].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert ds['qc_temp_mean'].attrs['flag_assessments'] == [
        'Bad', 'Bad', 'Bad', 'Indeterminate', 'Bad', 'Bad', 'Suspect', 'Good', 'Bad']
    assert ds['qc_temp_mean'].attrs['flag_meanings'][0] == 'Value is equal to missing_value.'
    assert ds['qc_temp_mean'].attrs['flag_meanings'][-1] == 'Values are bad for all'
    assert ds['qc_temp_mean'].attrs['flag_meanings'][-2] == 'Values are good'
    assert np.sum(ds['qc_temp_mean'].values) == 81344
    assert np.count_nonzero(ds['qc_temp_mean'].values) == 1423

    del ds

    ds = read_netcdf(EXAMPLE_MET1, keep_variables=['temp_mean', 'qc_temp_mean'], cleanup_qc=True)
    apply_supplemental_qc(ds, Path(EXAMPLE_MET_YAML).parent, apply_all=False)
    assert ds['qc_temp_mean'].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32, 64, 128]

    ds = read_netcdf(EXAMPLE_MET1, cleanup_qc=True)
    apply_supplemental_qc(ds, Path(EXAMPLE_MET_YAML).parent, exclude_all_variables='temp_mean')
    assert ds['qc_rh_mean'].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert 'Values are bad for all' in ds['qc_rh_mean'].attrs['flag_meanings']
    assert 'Values are bad for all' not in ds['qc_temp_mean'].attrs['flag_meanings']

    del ds

    ds = read_netcdf(EXAMPLE_MET1, keep_variables=['temp_mean', 'rh_mean'])
    apply_supplemental_qc(ds, Path(EXAMPLE_MET_YAML).parent, exclude_all_variables='temp_mean',
                          assessments='Bad', quiet=True)
    assert ds['qc_rh_mean'].attrs['flag_assessments'] == ['Bad']
    assert ds['qc_temp_mean'].attrs['flag_assessments'] == ['Bad', 'Bad']
    assert np.sum(ds['qc_rh_mean'].values) == 124
    assert np.sum(ds['qc_temp_mean'].values) == 2840

    del ds


def test_scalar_dqr():
    # Test DQR Webservice using known DQR
    ds = read_netcdf(EXAMPLE_ENA_MET)

    # DQR webservice does go down, so ensure it
    # properly runs first before testing
    try:
        ds = add_dqr_to_qc(ds)
        ran = True
    except ValueError:
        ran = False

    if ran:
        assert 'qc_lat' in ds
        assert np.size(ds['qc_lat'].values) == 1
        assert np.size(ds['qc_alt'].values) == 1
        assert np.size(ds['base_time'].values) == 1
