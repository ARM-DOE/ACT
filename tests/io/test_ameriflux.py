import act
import glob
import numpy as np
from numpy.testing import assert_equal
import xarray as xr
import pytest


def test_read_ameriflux():
    rename_vars_dict = {'FETCH_70': 'foo'}

    ds = act.io.read_ameriflux(
        act.tests.EXAMPLE_AMERIFLUX_BASE,
        metadata_filename=act.tests.EXAMPLE_AMERIFLUX_META,
        data_type='BASE',
        rename_vars_dict=rename_vars_dict,
    )
    # Test if variables are correct and units and fill values were added
    assert 'H2O' in ds.variables.keys()
    assert ds.variables['H2O'].attrs['units'] == 'mmolH2O mol-1'
    assert ds.variables['H2O'].attrs['_FILL_VALUE'] == -9999
    assert np.isnan(ds.variables['H2O'].values[0:5]).all()
    assert_equal(ds.variables['H2O'].values[-5:], [6.40095, 6.31573, 6.04937, 6.04294, 5.70889])

    assert 'T_SONIC' in ds.variables.keys()
    assert ds.variables['T_SONIC'].attrs['units'] == 'deg C'
    assert ds.variables['T_SONIC'].attrs['_FILL_VALUE'] == -9999
    assert np.isnan(ds.variables['T_SONIC'].values[0:5]).all()
    assert_equal(ds.variables['T_SONIC'].values[-5:], [4.30861, 4.19665, 4.16113, 3.99634, 3.86785])

    # Test if time is correct
    assert ds.time.values[0] == np.datetime64('2024-01-01T00:00:00.000000000')
    assert len(ds.time.values) == 17568

    # Test if renaming variables worked
    assert 'foo' in ds.variables.keys()

    # Test metadata
    assert ds.attrs['site'] == 'US-CU1'
    assert ds.attrs['version'] == '1-5'
    assert ds.attrs['SITE_NAME'] == 'UIC Plant Research Laboratory Chicago'

    assert len(ds.attrs['TEAM_MEMBERS']) == 9
    assert 'Bhupendra' in ds.attrs['TEAM_MEMBERS'][0]
    assert 'Matt' in ds.attrs['TEAM_MEMBERS'][6]
    assert 'TEAM_MEMBER_EMAIL' in ds.attrs['TEAM_MEMBERS'][0]
    assert 'TEAM_MEMBER_INSTITUTION' in ds.attrs['TEAM_MEMBERS'][0]
    assert 'TEAM_MEMBER_INSTITUTION:Argonne National Laboratory' in ds.attrs['TEAM_MEMBERS'][0]

    assert 'FLUX_MEASUREMENTS_VARIABLE:CO2' in ds.attrs['FLUX_MEASUREMENTS_METHODS'][0]
    assert 'FLUX_MEASUREMENTS_VARIABLE:H' in ds.attrs['FLUX_MEASUREMENTS_METHODS'][1]
    assert 'FLUX_MEASUREMENTS_METHOD:Eddy Covariance' in ds.attrs['FLUX_MEASUREMENTS_METHODS'][0]

    assert 'Bhupendra' in ds.attrs['DOI_CONTRIBUTORS'][0]
    assert 'Sujan' in ds.attrs['DOI_CONTRIBUTORS'][1]
    assert 'DOI_CONTRIBUTOR_DATAPRODUCT:AmeriFlux' in ds.attrs['DOI_CONTRIBUTORS'][0]
    assert 'DOI_CONTRIBUTOR_ROLE:Author' in ds.attrs['DOI_CONTRIBUTORS'][0]


def test_convert_to_ameriflux():
    files = glob.glob(act.tests.sample_files.EXAMPLE_ECORSF_E39)
    ds_ecor = act.io.arm.read_arm_netcdf(files)

    with pytest.warns(UserWarning, match="mapping was not provided"):
        df = act.io.ameriflux.convert_to_ameriflux(ds_ecor)

    assert 'FC' in df
    assert 'WS_MAX' in df

    files = glob.glob(act.tests.sample_files.EXAMPLE_SEBS_E39)
    ds_sebs = act.io.arm.read_arm_netcdf(files)

    ds = xr.merge([ds_ecor, ds_sebs], compat='override')
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
