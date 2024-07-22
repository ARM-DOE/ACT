import numpy as np
from os import environ
from pathlib import Path
import random
import pytest

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

        assert 'flag_masks' not in result[qc_var_name].attrs.keys()
        assert isinstance(result[qc_var_name].attrs['flag_values'], list)

        assert np.sum(result[qc_var_name].values) == 610

        qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Indeterminate')
        assert np.all(np.where(qc_ma.mask)[0] == np.arange(100, 170))

        qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
        index = np.concatenate([index_1, index_2, index_3])
        assert np.all(np.where(qc_ma.mask)[0] == index)

        assert "Quality control summary implemented by ACT" in result.attrs['history']


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

    result = ds.qcfilter.create_qc_summary()

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

    To Run this test set keyword on pytest command line:
    --runbig

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
    skip_datastream_codes = [
        'mmcrmom',
        'microbasepi',
        'lblch1a',
        'swats',
        '30co2flx4mmet',
        'microbasepi2',
        '30co2flx60m',
        'bbhrpavg1mlawer',
        'co',
        'lblch1b',
        '30co2flx25m',
        '30co2flx4m',
    ]
    num_files = 3
    testing_files = []
    expected_assessments = ['Not failing', 'Suspect', 'Indeterminate', 'Incorrect', 'Bad']

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

    for file in testing_files:
        print(f"Testing: {file}")
        ds = read_arm_netcdf(str(file), cleanup_qc=True)
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
            assert ds[qc_var_name].attrs['flag_meanings'][0] == 'Not failing quality control tests'

            for assessment in ds[qc_var_name].attrs['flag_assessments']:
                assert assessment in expected_assessments

        if created_qc_summary:
            assert "Quality control summary implemented by ACT" in ds.attrs['history']

        del ds
