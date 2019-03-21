import act
import glob


def test_io():
    sonde_ds = act.io.armfiles.read_netcdf(
        [act.tests.EXAMPLE_SONDE1])
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert sonde_ds.act.arm_standards_flag.OK
    sonde_ds.close()


def test_io_mfdataset():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_SONDE_WILDCARD)
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert len(sonde_ds.act.file_times) == 7
    assert sonde_ds.act.arm_standards_flag.OK
    sonde_ds.close()
