import act
import glob
import xarray as xr
import pytest


def test_convert_to_ameriflux():
    files = glob.glob(act.tests.sample_files.EXAMPLE_ECORSF_E39)
    ds_ecor = act.io.arm.read_arm_netcdf(files)

    with pytest.warns(UserWarning, match="mapping was not provided"):
        df = act.io.ameriflux.convert_to_ameriflux(ds_ecor)

    assert 'FC' in df
    assert 'WS_MAX' in df

    files = glob.glob(act.tests.sample_files.EXAMPLE_SEBS_E39)
    ds_sebs = act.io.arm.read_arm_netcdf(files)

    ds = xr.merge([ds_ecor, ds_sebs])
    with pytest.warns(UserWarning, match="mapping was not provided"):
        df = act.io.ameriflux.convert_to_ameriflux(ds)

    assert 'SWC_2_1_1' in df
    assert 'TS_3_1_1' in df
    assert 'G_2_1_1' in df

    files = glob.glob(act.tests.sample_files.EXAMPLE_STAMP_E39)
    ds_stamp = act.io.arm.read_arm_netcdf(files)

    ds = xr.merge([ds_ecor, ds_sebs, ds_stamp], compat='override')
    with pytest.warns(UserWarning, match="mapping was not provided"):
        df = act.io.ameriflux.convert_to_ameriflux(ds)

    assert 'SWC_6_10_1' in df
    assert 'G_2_1_1' in df
    assert 'TS_5_2_1' in df
