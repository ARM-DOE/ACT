import pytest

import act


def test_decode_present_weather():
    ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_MET1)
    ds = act.utils.decode_present_weather(ds, variable='pwd_pw_code_inst')

    data = ds['pwd_pw_code_inst_decoded'].values
    result = 'No significant weather observed'
    assert data[0] == result
    assert data[100] == result
    assert data[600] == result

    pytest.raises(ValueError, act.utils.inst_utils.decode_present_weather, ds)
    pytest.raises(
        ValueError,
        act.utils.inst_utils.decode_present_weather,
        ds,
        variable='temp_temp',
    )
