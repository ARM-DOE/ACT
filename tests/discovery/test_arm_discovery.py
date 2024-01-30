import glob
import os

import numpy as np

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
