from pathlib import Path

import numpy as np

from act.io.arm import read_arm_netcdf
from act.qc.add_supplemental_qc import apply_supplemental_qc, read_yaml_supplemental_qc
from act.tests import EXAMPLE_MET1, EXAMPLE_MET_YAML


def test_read_yaml_supplemental_qc():
    ds = read_arm_netcdf(
        EXAMPLE_MET1, keep_variables=['temp_mean', 'qc_temp_mean'], cleanup_qc=True
    )

    result = read_yaml_supplemental_qc(ds, EXAMPLE_MET_YAML)
    assert isinstance(result, dict)
    assert len(result.keys()) == 3

    result = read_yaml_supplemental_qc(
        ds,
        Path(EXAMPLE_MET_YAML).parent,
        variables='temp_mean',
        assessments=['Bad', 'Incorrect', 'Suspect'],
    )
    assert len(result.keys()) == 2
    assert sorted(result['temp_mean'].keys()) == ['Bad', 'Suspect']

    result = read_yaml_supplemental_qc(ds, 'sgpmetE13.b1.yaml', quiet=True)
    assert result is None

    apply_supplemental_qc(ds, EXAMPLE_MET_YAML)
    assert ds['qc_temp_mean'].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert ds['qc_temp_mean'].attrs['flag_assessments'] == [
        'Bad',
        'Bad',
        'Bad',
        'Indeterminate',
        'Bad',
        'Bad',
        'Suspect',
        'Good',
        'Bad',
    ]
    assert ds['qc_temp_mean'].attrs['flag_meanings'][0] == 'Value is equal to missing_value.'
    assert ds['qc_temp_mean'].attrs['flag_meanings'][-1] == 'Values are bad for all'
    assert ds['qc_temp_mean'].attrs['flag_meanings'][-2] == 'Values are good'
    assert np.sum(ds['qc_temp_mean'].values) == 81344
    assert np.count_nonzero(ds['qc_temp_mean'].values) == 1423

    del ds

    ds = read_arm_netcdf(
        EXAMPLE_MET1, keep_variables=['temp_mean', 'qc_temp_mean'], cleanup_qc=True
    )
    apply_supplemental_qc(ds, Path(EXAMPLE_MET_YAML).parent, apply_all=False)
    assert ds['qc_temp_mean'].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32, 64, 128]

    ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=True)
    apply_supplemental_qc(ds, Path(EXAMPLE_MET_YAML).parent, exclude_all_variables='temp_mean')
    assert ds['qc_rh_mean'].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32, 64, 128]
    assert 'Values are bad for all' in ds['qc_rh_mean'].attrs['flag_meanings']
    assert 'Values are bad for all' not in ds['qc_temp_mean'].attrs['flag_meanings']

    del ds

    ds = read_arm_netcdf(EXAMPLE_MET1, keep_variables=['temp_mean', 'rh_mean'])
    apply_supplemental_qc(
        ds,
        Path(EXAMPLE_MET_YAML).parent,
        exclude_all_variables='temp_mean',
        assessments='Bad',
        quiet=True,
    )
    assert ds['qc_rh_mean'].attrs['flag_assessments'] == ['Bad']
    assert ds['qc_temp_mean'].attrs['flag_assessments'] == ['Bad', 'Bad']
    assert np.sum(ds['qc_rh_mean'].values) == 124
    assert np.sum(ds['qc_temp_mean'].values) == 2840

    del ds
