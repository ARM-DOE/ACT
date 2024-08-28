import numpy as np
from os import environ
from pathlib import Path
import random
import pytest
import datetime

from act.io.arm import read_arm_netcdf
from act.tests import EXAMPLE_MET1
from act.qc.qcfilter import set_bit
from act.utils.data_utils import DatastreamParserARM


def test_qc_summary():
    for cleanup in [False, True]:
        ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=not cleanup)
        for var_name in ['temp_mean', 'rh_mean']:
            qc_var_name = f'qc_{var_name}'
            qc_data = ds[qc_var_name].values

            assert np.sum(qc_data) == 0

            index_4 = np.arange(100, 200)
            qc_data[index_4] = set_bit(qc_data[index_4], 4)
            index_1 = np.arange(170, 230)
            qc_data[index_1] = set_bit(qc_data[index_1], 1)
            index_2 = np.arange(250, 400)
            qc_data[index_2] = set_bit(qc_data[index_2], 2)
            index_3 = np.arange(450, 510)
            qc_data[index_3] = set_bit(qc_data[index_3], 3)
            ds[qc_var_name].values = qc_data

        result = ds.qcfilter.create_qc_summary(cleanup_qc=cleanup)

        for var_name in ['temp_mean', 'rh_mean']:
            assert 'flag_masks' not in result[qc_var_name].attrs.keys()
            assert isinstance(result[qc_var_name].attrs['flag_values'], list)

            assert np.sum(result[qc_var_name].values) == 880

            qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Suspect')
            assert np.sum(np.where(qc_ma.mask)) == 9415

            qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Incorrect')
            assert np.sum(np.where(qc_ma.mask)) == 89415

            att_names = [
                'fail_min',
                'fail_max',
                'fail_delta',
                'valid_min',
                'valid_max',
                'valid_delta',
            ]
            for att_name in att_names:
                assert att_name not in ds[f'qc_{var_name}'].attrs

        assert "Quality control summary implemented by ACT" in result.attrs['history']

        del ds


def test_qc_summary_multiple_assessment_names():
    ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=True)
    var_name = 'temp_mean'
    qc_var_name = f'qc_{var_name}'
    qc_data = ds[qc_var_name].values

    assert np.sum(qc_data) == 0

    index_4 = np.arange(200, 300)
    qc_data[index_4] = set_bit(qc_data[index_4], 4)
    index_1 = np.arange(270, 330)
    qc_data[index_1] = set_bit(qc_data[index_1], 1)
    index_2 = np.arange(350, 500)
    qc_data[index_2] = set_bit(qc_data[index_2], 2)
    index_3 = np.arange(550, 610)
    qc_data[index_3] = set_bit(qc_data[index_3], 3)
    ds[qc_var_name].values = qc_data

    index_5 = np.arange(50, 150)
    ds.qcfilter.add_test(
        var_name, index=index_5, test_meaning='Testing Suspect', test_assessment='Suspect'
    )

    index_6 = np.arange(130, 210)
    ds.qcfilter.add_test(
        var_name, index=index_6, test_meaning='Testing Incorrect', test_assessment='Incorrect'
    )

    result = ds.qcfilter.create_qc_summary(normalize_assessment=False)

    assert result[qc_var_name].attrs['flag_assessments'] == [
        'Not failing',
        'Suspect',
        'Indeterminate',
        'Incorrect',
        'Bad',
    ]

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Indeterminate')
    assert np.sum(np.where(qc_ma.mask)[0]) == 14370

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Suspect')
    assert np.sum(np.where(qc_ma.mask)[0]) == 7160

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
    assert np.sum(np.where(qc_ma.mask)[0]) == 116415

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Incorrect')
    assert np.sum(np.where(qc_ma.mask)[0]) == 13560

    assert np.sum(np.where(result[qc_var_name].values == 0)) == 884575
    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Not failing')
    assert np.sum(np.where(qc_ma.mask)[0]) == 884575


def test_qc_summary_unexpected_assessment_name():
    var_name = 'temp_mean'
    ds = read_arm_netcdf(EXAMPLE_MET1, keep_variables=var_name)

    test_meanings = [
        'Testing Bad',
        'Testing Boomer',
        'Testing Boomer Second',
        'Testing Incorrect',
        'Testing Indeterminate',
        'Testing Sooner',
        'Testing Suspect',
    ]
    test_assessments = [
        'Bad',
        'Boomer',
        'boomer',
        'Incorrect',
        'Indeterminate',
        'Sooner',
        'Suspect',
    ]

    test_index_sums = [4950, 39900, 39900, 34950, 44950, 54950, 64950]

    for ii, _ in enumerate(test_assessments):
        ds.qcfilter.add_test(
            var_name,
            index=np.arange(ii * 100, ii * 100 + 100),
            test_meaning=test_meanings[ii],
            test_assessment=test_assessments[ii],
        )

    ds = ds.qcfilter.create_qc_summary(normalize_assessment=False)

    qc_var_name = ds.qcfilter.check_for_ancillary_qc(var_name, add_if_missing=False)

    # Make sure flag meanings are correct with new assessments.
    assert sorted(ds[qc_var_name].attrs['flag_meanings']) == [
        'Data Boomer',
        'Data Sooner',
        'Data incorrect use not recommended',
        'Data incorrect use not recommended',
        'Data suspect further analysis recommended',
        'Data suspect further analysis recommended',
        'Not failing quality control tests',
    ]
    assert sorted(ds[qc_var_name].attrs['flag_assessments']) == [
        'Bad',
        'Boomer',
        'Incorrect',
        'Indeterminate',
        'Not failing',
        'Sooner',
        'Suspect',
    ]
    # Make sure the values and order of first 5 are as expected. The other non-standard
    # assessments may be in different order with set operations.
    assert ds[qc_var_name].attrs['flag_assessments'][:5] == [
        'Not failing',
        'Suspect',
        'Indeterminate',
        'Incorrect',
        'Bad',
    ]

    for assessment, index_sum in zip(test_assessments, test_index_sums):
        qc_ma = ds.qcfilter.get_masked_data(var_name, rm_assessments=assessment)
        assert np.sum(np.where(qc_ma.mask)[0]) == index_sum

    qc_ma = ds.qcfilter.get_masked_data(var_name, rm_assessments=['Bucky'])
    assert np.sum(np.where(qc_ma.mask)[0]) == 0

    qc_ma = ds.qcfilter.get_masked_data(var_name, rm_assessments=['Boomer', 'Sooner'])
    assert np.sum(np.where(qc_ma.mask)[0]) == 94850

    qc_ma = ds.qcfilter.get_masked_data(
        var_name,
        rm_assessments=['Boomer', 'Sooner', 'Indeterminate', 'Suspect', 'Bad', 'Incorrect'],
    )
    assert np.sum(np.where(qc_ma.mask)[0]) == 244650

    del ds


def test_qc_summary_scalar():
    # Test scalar variables. Currently not implemented so just check that we
    # don't do anything.
    var_names = ['alt', 'temp_mean']
    ds = read_arm_netcdf(EXAMPLE_MET1, keep_variables=var_names)

    test_meanings = ['Testing Incorrect', 'Testing Suspect']
    test_assessments = ['Incorrect', 'Suspect']

    for var_name in var_names:
        for ii, _ in enumerate(test_assessments):
            ds.qcfilter.add_test(
                var_name,
                index=0,
                test_meaning=test_meanings[ii],
                test_assessment=test_assessments[ii],
            )

    with pytest.warns(UserWarning, match="Unable to process scalar variable"):
        ds = ds.qcfilter.create_qc_summary(normalize_assessment=False)

    assert 'flag_masks' in ds[f'qc_{var_names[0]}'].attrs.keys()
    assert 'flag_values' not in ds[f'qc_{var_names[0]}'].attrs.keys()
    assert 'flag_masks' not in ds[f'qc_{var_names[1]}'].attrs.keys()
    assert 'flag_values' in ds[f'qc_{var_names[1]}'].attrs.keys()


@pytest.mark.big
@pytest.mark.skipif('ARCHIVE_DATA' not in environ, reason="Running outside ADC system.")
def test_qc_summary_big_data():
    """
    We want to test on as much ARM data as possible. But we do not want to force
    a large amount of test data in GitHub. Plan is to see if the pytest code is being
    run on ARM system and if so then run on historical data. If running on GitHub
    then don't run tests. Also, have a switch to not force this big test to always
    run as that would be mean to the developer. So need to periodicaly run with the
    manual switch enabled.

    All exceptions are caught and a file name is sent to the output file when
    an exception is found. Since this is testing 10,000+ files it will take hours
    to run. I suggest you run in background and capture the standard out to a different
    file. If no files are written to the output file then all tests passed.

    Output file name follows the convention of:
        ~/test_qc_summary_big_data.{datetime}.txt

    To Run this test set keyword on pytest command line:
    > pytest -s --runbig test_qc_summary.py::test_qc_summary_big_data &> ~/out.txt &


    """

    base_path = Path(environ['ARCHIVE_DATA'])
    if not base_path.is_dir():
        return

    # Set number of files from each directory to test.
    skip_sites = [
        'shb',
        'wbu',
        'dna',
        'rld',
        'smt',
        'nic',
        'isp',
        'dmf',
        'nac',
        'rev',
        'yeu',
        'zrh',
        'osc',
    ]
    skip_datastream_codes = ['mmcrmom']
    num_files = 3
    expected_assessments = ['Not failing', 'Suspect', 'Indeterminate', 'Incorrect', 'Bad']

    testing_files = []

    if len(testing_files) == 0:
        filename = (
            f'test_qc_summary_big_data.{datetime.datetime.utcnow().strftime("%Y%m%d.%H%M%S")}.txt'
        )
        output_file = Path(environ['HOME'], filename)
        output_file.unlink(missing_ok=True)
        output_file.touch()

        site_dirs = list(base_path.glob('???'))
        for site_dir in site_dirs:
            if site_dir.name in skip_sites:
                continue

            datastream_dirs = list(site_dir.glob('*.[bc]?'))
            for datastream_dir in datastream_dirs:
                if '-' in datastream_dir.name:
                    continue

                fn_obj = DatastreamParserARM(datastream_dir.name)
                facility = fn_obj.facility
                if facility is not None and facility[0] in ['A', 'X', 'U', 'F', 'N']:
                    continue

                datastream_class = fn_obj.datastream_class
                if datastream_class is not None and datastream_class in skip_datastream_codes:
                    continue

                files = list(datastream_dir.glob('*.nc'))
                files.extend(datastream_dir.glob('*.cdf'))
                if len(files) == 0:
                    continue

                num_tests = num_files
                if len(files) < num_files:
                    num_tests = len(files)

                for ii in range(0, num_tests):
                    testing_files.append(random.choice(files))

        print(f"\nTesting {len(testing_files)} files\n")
        print(f"Output file name = {output_file}\n")

    for file in testing_files:
        try:
            print(f"Testing: {file}")
            ds = read_arm_netcdf(str(file), cleanup_qc=True, decode_times=False)
            ds = ds.qcfilter.create_qc_summary()

            created_qc_summary = False
            for var_name in ds.data_vars:
                qc_var_name = ds.qcfilter.check_for_ancillary_qc(
                    var_name, add_if_missing=False, cleanup=False
                )

                if qc_var_name is None:
                    continue

                created_qc_summary = True
                assert isinstance(ds[qc_var_name].attrs['flag_values'], list)
                assert isinstance(ds[qc_var_name].attrs['flag_assessments'], list)
                assert isinstance(ds[qc_var_name].attrs['flag_meanings'], list)
                assert len(ds[qc_var_name].attrs['flag_values']) >= 1
                assert len(ds[qc_var_name].attrs['flag_assessments']) >= 1
                assert len(ds[qc_var_name].attrs['flag_meanings']) >= 1
                assert ds[qc_var_name].attrs['flag_assessments'][0] == 'Not failing'
                assert (
                    ds[qc_var_name].attrs['flag_meanings'][0] == 'Not failing quality control tests'
                )

                for assessment in ds[qc_var_name].attrs['flag_assessments']:
                    assert assessment in expected_assessments

            if created_qc_summary:
                assert "Quality control summary implemented by ACT" in ds.attrs['history']

            del ds

        except Exception:
            with open(output_file, "a") as myfile:
                myfile.write(f"{file}\n")
