import glob

import act


def test_read_neon():
    data_file = glob.glob(act.tests.EXAMPLE_NEON)
    variable_file = glob.glob(act.tests.EXAMPLE_NEON_VARIABLE)
    position_file = glob.glob(act.tests.EXAMPLE_NEON_POSITION)

    ds = act.io.neon.read_neon_csv(data_file)
    assert len(ds['time'].values) == 17280
    assert 'time' in ds
    assert 'tempSingleMean' in ds
    assert ds['tempSingleMean'].values[0] == -0.6003

    ds = act.io.neon.read_neon_csv(
        data_file, variable_files=variable_file, position_files=position_file
    )
    assert ds['northOffset'].values == -5.79
    assert ds['tempSingleMean'].attrs['units'] == 'celsius'
    assert 'lat' in ds
    assert ds['lat'].values == 71.282425
