import numpy as np

from act.io.arm import read_arm_netcdf
from act.tests import (
    EXAMPLE_CEIL1,
    EXAMPLE_CO2FLX4M,
    EXAMPLE_MET1,
    EXAMPLE_MET_SAIL,
    EXAMPLE_SIRS_SIRI_QC,
    EXAMPLE_SWATS,
)


def test_global_qc_cleanup():
    ds = read_arm_netcdf(EXAMPLE_MET1)
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


def test_clean():
    # Read test data
    ceil_ds = read_arm_netcdf([EXAMPLE_CEIL1])
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


def test_qc_remainder():
    ds = read_arm_netcdf(EXAMPLE_MET1)
    assert ds.clean.get_attr_info(variable='bad_name') is None
    del ds.attrs['qc_bit_comment']
    assert isinstance(ds.clean.get_attr_info(), dict)
    ds.attrs['qc_flag_comment'] = 'testing'
    ds.close()

    ds = read_arm_netcdf(EXAMPLE_MET1)
    ds.clean.cleanup(normalize_assessment=True)
    ds['qc_atmos_pressure'].attrs['units'] = 'testing'
    del ds['qc_temp_mean'].attrs['units']
    del ds['qc_temp_mean'].attrs['flag_masks']
    ds.clean.handle_missing_values()
    ds.close()

    ds = read_arm_netcdf(EXAMPLE_MET1)
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

    ds = read_arm_netcdf(EXAMPLE_CO2FLX4M)
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


def test_clean_sirs_siri_qc():
    ds = read_arm_netcdf(EXAMPLE_SIRS_SIRI_QC)

    data = ds["qc_short_direct_normal"].values
    data[0:5] = 1
    data[5:10] = 2
    data[10:15] = 3
    data[15:20] = 6
    data[20:25] = 7
    data[25:30] = 8
    data[30:35] = 9
    data[35:40] = 94
    data[40:45] = 95
    data[45:50] = 96
    data[50:55] = 97
    data[55:60] = 14
    data[60:65] = 18
    data[65:70] = 22
    data[70:75] = 26
    ds["qc_short_direct_normal"].values = data

    data = ds["qc_up_long_hemisp"].values
    data[0:5] = 1
    data[5:10] = 2
    data[10:15] = 7
    data[15:20] = 8
    data[20:25] = 31
    ds["qc_up_long_hemisp"].values = data

    data = ds["qc_up_short_hemisp"].values
    data[0:5] = 1
    data[5:10] = 2
    data[10:15] = 7
    data[15:20] = 8
    data[20:25] = 31
    ds["qc_up_short_hemisp"].values = data

    ds.clean.cleanup()

    assert ds["qc_short_direct_normal"].attrs['flag_masks'] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    ]
    assert ds["qc_short_direct_normal"].attrs['flag_meanings'] == [
        'Value is set to missing_value.',
        'Passed 1-component test; data fall within max-min limits of Kt,Kn, or Kd',
        'Passed 2-component test; data fall within 0.03 of the Gompertz boundaries',
        'Passed 3-component test; data come within +/- 0.03 of satifying Kt=Kn+Kd',
        'Value estimated; passes all pertinent SERI QC tests',
        'Failed 1-component test; lower than allowed minimum',
        'Falied 1-component test; higher than allowed maximum',
        'Passed 3-component test but failed 2-component test by >0.05',
        'Data fall into a physically impossible region where Kn>Kt by K-space distances of 0.05 to 0.10.',
        'Data fall into a physically impossible region where Kn>Kt by K-space distances of 0.10 to 0.15.',
        'Data fall into a physically impossible region where Kn>Kt by K-space distances of 0.15 to 0.20.',
        'Data fall into a physically impossible region where Kn>Kt by K-space distances of >= 0.20.',
        'Parameter too low by 3-component test (Kt=Kn+Kd)',
        'Parameter too high by 3-component test (Kt=Kn+Kd)',
        'Parameter too low by 2-component test (Gompertz boundary)',
        'Parameter too high by 2-component test (Gompertz boundary)',
    ]

    assert ds["qc_up_long_hemisp"].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32]
    assert ds["qc_up_long_hemisp"].attrs['flag_meanings'] == [
        'Value is set to missing_value.',
        'Passed 1-component test; data fall within max-min limits of up_long_hemisp and down_long_hemisp_shaded, but short_direct_normal and down_short_hemisp or down_short_diffuse fail the SERI QC tests.',
        'Passed 2-component test; data fall within max-min limits of up_long_hemisp and down_long_hemisp_shaded, and short_direct_normal, or down_short_hemisp and down_short_diffuse pass the SERI QC tests while the difference between down_short_hemisp and down_short_diffuse is greater than 20 W/m2.',
        'Failed 1-component test; lower than allowed minimum',
        'Failed 1-component test; higher than allowed maximum',
        'Failed 2-component test',
    ]

    assert ds["qc_up_short_hemisp"].attrs['flag_masks'] == [1, 2, 4, 8, 16, 32]
    assert ds["qc_up_short_hemisp"].attrs['flag_meanings'] == [
        'Value is set to missing_value.',
        'Passed 1-component test',
        'Passed 2-component test',
        'Failed 1-component test; lower than allowed minimum',
        'Failed 1-component test; higher than allowed maximum',
        'Failed 2-component test; solar zenith angle is less than 80 degrees and down_short_hemisp is 0 or missing',
    ]

    assert np.all(ds["qc_short_direct_normal"].values[0:5] == 2)
    assert np.all(ds["qc_short_direct_normal"].values[5:10] == 4)
    assert np.all(ds["qc_short_direct_normal"].values[10:15] == 8)
    assert np.all(ds["qc_short_direct_normal"].values[15:20] == 16)
    assert np.all(ds["qc_short_direct_normal"].values[20:25] == 32)
    assert np.all(ds["qc_short_direct_normal"].values[25:30] == 64)
    assert np.all(ds["qc_short_direct_normal"].values[30:35] == 128)
    assert np.all(ds["qc_short_direct_normal"].values[35:40] == 256)
    assert np.all(ds["qc_short_direct_normal"].values[40:45] == 512)
    assert np.all(ds["qc_short_direct_normal"].values[45:50] == 1024)
    assert np.all(ds["qc_short_direct_normal"].values[50:55] == 2048)
    assert np.all(ds["qc_short_direct_normal"].values[55:60] == 4096)
    assert np.all(ds["qc_short_direct_normal"].values[60:65] == 8192)
    assert np.all(ds["qc_short_direct_normal"].values[65:70] == 16384)
    assert np.all(ds["qc_short_direct_normal"].values[70:75] == 32768)

    assert np.all(ds["qc_up_long_hemisp"].values[0:5] == 2)
    assert np.all(ds["qc_up_long_hemisp"].values[5:10] == 4)
    assert np.all(ds["qc_up_long_hemisp"].values[10:15] == 8)
    assert np.all(ds["qc_up_long_hemisp"].values[15:20] == 16)
    assert np.all(ds["qc_up_long_hemisp"].values[20:25] == 32)

    assert np.all(ds["qc_up_short_hemisp"].values[0:5] == 2)
    assert np.all(ds["qc_up_short_hemisp"].values[5:10] == 4)
    assert np.all(ds["qc_up_short_hemisp"].values[10:15] == 8)
    assert np.all(ds["qc_up_short_hemisp"].values[15:20] == 16)
    assert np.all(ds["qc_up_short_hemisp"].values[20:25] == 32)


def test_swats_qc():
    ds = read_arm_netcdf(EXAMPLE_SWATS)
    ds.clean.cleanup()

    data_var_names = []
    for var_name in ds.data_vars:
        try:
            ds[f'qc_{var_name}']
            data_var_names.append(var_name)
        except KeyError:
            pass

    for var_name in data_var_names:
        qc_var_name = f'qc_{var_name}'

        assert ds[qc_var_name].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds[qc_var_name].attrs['flag_meanings'] == [
            'Value is set to missing_value.',
            'Data value less than fail_min.',
            'Data value greater than fail_max.',
            'Difference between current and previous values exceeds fail_delta.',
        ]
        assert ds[qc_var_name].attrs['flag_assessments'] == ['Bad', 'Bad', 'Bad', 'Indeterminate']
        assert 'fail_min' in ds[qc_var_name].attrs
        assert 'fail_max' in ds[qc_var_name].attrs
        assert 'fail_delta' in ds[qc_var_name].attrs

        assert 'valid_min' not in ds[var_name].attrs
        assert 'valid_max' not in ds[var_name].attrs
        assert 'valid_delta' not in ds[var_name].attrs
        assert ds[var_name].attrs['units'] != 'C'


def test_fix_incorrect_variable_bit_description_attributes():
    ds = read_arm_netcdf(EXAMPLE_MET_SAIL)
    qc_var_name = 'qc_temp_mean'
    ds[qc_var_name].attrs['qc_bit_2_description'] = ds[qc_var_name].attrs['bit_2_description']
    ds[qc_var_name].attrs['qc_bit_2_assessment'] = ds[qc_var_name].attrs['bit_2_assessment']
    del ds[qc_var_name].attrs['bit_2_description']
    del ds[qc_var_name].attrs['bit_2_assessment']

    ds.clean.cleanup()

    assert ds[qc_var_name].attrs['flag_masks'] == [1, 2, 4, 8]
    assert ds[qc_var_name].attrs['flag_meanings'] == [
        'Value is equal to missing_value.',
        'Value is less than fail_min.',
        'Value is greater than fail_max.',
        'Difference between current and previous values exceeds fail_delta.',
    ]
    assert ds[qc_var_name].attrs['flag_assessments'] == ['Bad', 'Bad', 'Bad', 'Indeterminate']
    assert 'qc_bit_2_description' not in ds[qc_var_name].attrs
    assert 'qc_bit_2_assessment' not in ds[qc_var_name].attrs
