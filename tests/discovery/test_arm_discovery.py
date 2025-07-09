import glob
import os

import numpy as np
import pytest

import act


def test_download_armdata():
    if not os.path.isdir(os.getcwd() + '/data/'):
        os.makedirs(os.getcwd() + '/data/')

    # Place your username and token here
    username = os.getenv('ARM_USERNAME')
    token = os.getenv('ARM_PASSWORD')

    if username is not None and token is not None:
        if len(username) == 0 and len(token) == 0:
            return
        datastream = 'sgpmetE13.b1'
        startdate = '2020-01-01'
        enddate = startdate
        outdir = os.getcwd() + '/data/'

        results = act.discovery.arm.download_arm_data(
            username, token, datastream, startdate, enddate, output=outdir
        )
        files = glob.glob(outdir + datastream + '*20200101*cdf')
        if len(results) > 0:
            assert files is not None
            assert 'sgpmetE13' in files[0]

        if files is not None:
            if len(files) > 0:
                os.remove(files[0])

        datastream = 'sgpmeetE13.b1'
        act.discovery.arm.download_arm_data(
            username, token, datastream, startdate, enddate, output=outdir
        )
        files = glob.glob(outdir + datastream + '*20200101*cdf')
        assert len(files) == 0

        with np.testing.assert_raises(ConnectionRefusedError):
            act.discovery.arm.download_arm_data(
                username, token + '1234', datastream, startdate, enddate, output=outdir
            )

        datastream = 'sgpmetE13.b1'
        results = act.discovery.arm.download_arm_data(
            username, token, datastream, startdate, enddate
        )
        assert len(results) == 1


def test_download_armdata_hourly():
    if not os.path.isdir(os.getcwd() + '/data/'):
        os.makedirs(os.getcwd() + '/data/')

    # Place your username and token here
    username = os.getenv('ARM_USERNAME')
    token = os.getenv('ARM_PASSWORD')

    if username is not None and token is not None:
        if len(username) == 0 and len(token) == 0:
            return
        datastream = 'sgpmetE13.b1'
        startdate = '2020-01-01T00:00:00'
        enddate = '2020-01-01T12:00:00'
        outdir = os.getcwd() + '/data/'

        results = act.discovery.arm.download_arm_data(
            username, token, datastream, startdate, enddate, output=outdir
        )
        files = glob.glob(outdir + datastream + '*20200101*cdf')
        if len(results) > 0:
            assert files is not None
            assert 'sgpmetE13' in files[0]

        if files is not None:
            if len(files) > 0:
                os.remove(files[0])

        datastream = 'sgpmeetE13.b1'
        act.discovery.arm.download_arm_data(
            username, token, datastream, startdate, enddate, output=outdir
        )
        files = glob.glob(outdir + datastream + '*20200101*cdf')
        assert len(files) == 0

        with np.testing.assert_raises(ConnectionRefusedError):
            act.discovery.arm.download_arm_data(
                username, token + '1234', datastream, startdate, enddate, output=outdir
            )

        datastream = 'sgpmetE13.b1'
        results = act.discovery.arm.download_arm_data(
            username, token, datastream, startdate, enddate
        )
        assert len(results) == 1


def test_download_arm_data_mod_with_variables():
    # Call the function
    username = os.getenv('ARM_USERNAME')
    token = os.getenv('ARM_PASSWORD')
    if username is not None and token is not None:
        if len(username) == 0 and len(token) == 0:
            return

        file_list = [
            "sgpmetE13.b1.20200101.000000.cdf",
            "sgpmetE13.b1.20200102.000000.cdf",
        ]
        variables = [
            "temp_mean",
            "rh_mean",
        ]

        act.discovery.download_arm_data_mod(username, token, file_list, variables=variables)
        # Check if the file was saved correctly
        assert os.path.exists('sgpmetE13.b1.20200101-20200102.cdf')

        # Check if variables are only what was selected
        ds = act.io.read_arm_netcdf('sgpmetE13.b1.20200101-20200102.cdf')
        assert len(ds.variables.keys()) == 3

        # Check if all variables are present
        act.discovery.download_arm_data_mod(username, token, file_list)
        ds = act.io.read_arm_netcdf('sgpmetE13.b1.20200101-20200102.cdf')
        assert len(ds.variables.keys()) == 50

        # Check if custom output path exist and directory is created.
        act.discovery.download_arm_data_mod(
            username, token, file_list, output=os.getcwd() + '/data/'
        )
        ds = act.io.read_arm_netcdf(os.getcwd() + '/data/' + 'sgpmetE13.b1.20200101-20200102.cdf')
        assert len(ds.variables.keys()) == 50


def test_download_arm_data_mod_csv():
    username = os.getenv('ARM_USERNAME')
    token = os.getenv('ARM_PASSWORD')
    if username is not None and token is not None:
        if len(username) == 0 and len(token) == 0:
            return

        file_list = [
            "sgpmetE13.b1.20200101.000000.cdf",
            "sgpmetE13.b1.20200102.000000.cdf",
        ]
        # Test if csv file is created
        act.discovery.download_arm_data_mod(username, token, file_list, filetype='csv')

        # Check if the file was saved correctly
        assert os.path.exists('sgpmetE13.b1.20200101-20200102.csv')


def test_download_arm_data_mod_warn():
    # Check for warning if parameters are wrong
    with pytest.warns(UserWarning):
        username = 'foo'
        token = 'bar'
        file_list = [
            "sgpmetE13.b1.20200101.000000.cdf",
            "sgpmetE13.b1.20200102.000000.cdf",
        ]
        act.discovery.download_arm_data_mod(username, token, file_list, filetype='csv')


def test_arm_doi():
    datastream = 'sgpmetE13.b1'
    startdate = '2022-01-01'
    enddate = '2022-12-31'
    doi = act.discovery.get_arm_doi(datastream, startdate, enddate)

    assert len(doi) > 10
    assert isinstance(doi, str)
    assert 'doi' in doi
    assert 'Kyrouac' in doi

    doi = act.discovery.get_arm_doi('test', startdate, enddate)
    assert 'No DOI Found' in doi
