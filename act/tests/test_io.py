import act


def test_io():
    sonde_ds = act.io.armfiles.read_netcdf(
        [act.tests.EXAMPLE_MET1])
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert sonde_ds.act.arm_standards_flag.OK
    sonde_ds.close()


def test_io_mfdataset():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET_WILDCARD)
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert len(sonde_ds.act.file_times) == 7
    assert sonde_ds.act.arm_standards_flag.OK
    sonde_ds.close()

def test_io_anl_csv():
    anl_ds = act.io.csvfiles.read_csv(
        act.tests.EXAMPLE_ANL_CSV)
    assert 'temp_60m' in anl_ds.variables.keys()
    assert 'rh' in anl_ds.variables.keys()
    assert anl_ds.act.arm_standards_flag.OK
    anl_ds.close()
