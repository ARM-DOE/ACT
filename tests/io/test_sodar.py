import fsspec
import numpy as np

import act


def test_read_sodar():
    ds = act.io.read_mfas_sodar(act.tests.EXAMPLE_MFAS_SODAR)

    # Test coordinates.
    assert ds.time.shape[0] == 96
    assert ds.time[0].dtype == 'datetime64[ns]'

    assert ds.height.shape[0] == 58
    assert ds.height[0] == 30.0
    assert ds.height[-1] == 600.0

    # Test variable data, shape and attributes.
    assert len(ds.data_vars) == 26
    assert ds['dir'].shape == (96, 58)
    direction = ds['dir'][0, 0:5].values
    np.testing.assert_allclose(direction, [129.9, 144.2, 147.5, 143.5, 143.0], rtol=1e-6)
    pgz = ds['PGz'][0, 0:5].values
    np.testing.assert_allclose(pgz, [4, 4, 4, 5, 5])

    assert ds['dir'].attrs['variable_name'] == 'wind direction'
    assert ds['dir'].attrs['symbol'] == 'dir'
    assert ds['dir'].attrs['type'] == 'R1'
    assert ds['dir'].attrs['_FillValue'] == 999.9
    assert ds['dir'].attrs['error_mask'] == '0'
    assert ds['dir'].attrs['units'] == 'deg'

    # Test global attributes.
    assert ds.attrs['height above sea level [m]'] == 0.0
    assert ds.attrs['instrument_type'] == 'MFAS'


def test_metadata_retrieval():
    # Read the file and lines.
    file = fsspec.open(act.tests.EXAMPLE_MFAS_SODAR).open()
    lines = file.readlines()
    lines = [x.decode().rstrip()[:] for x in lines]

    # Retrieve metadata.
    file_dict, variable_dict = act.io.sodar._metadata_retrieval(lines)

    # Test file dictionary.
    assert 'instrument_type' in file_dict
    assert 'height above sea level [m]' in file_dict

    assert file_dict['format'] == 'FORMAT-1'
    assert file_dict['height above ground [m]'] == 0.0

    # Test variable dictionary.
    assert 'speed' in variable_dict.keys()
    assert 'error' not in variable_dict.keys()

    assert 'variable_name' in variable_dict['sigSpeed']
    assert 'units' in variable_dict['sigSpeed']
    assert '_FillValue' in variable_dict['sigSpeed']

    assert variable_dict['W']['units'] == 'm/s'
    assert variable_dict['W']['variable_name'] == 'wind W'
    assert variable_dict['W']['_FillValue'] == 99.99
