import act


def test_io():
    sonde_ds = act.io.armfiles.read_netcdf(
        [act.tests.EXAMPLE_MET1])
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert sonde_ds.attrs['_arm_standards_flag'].OK
    sonde_ds.close()


def test_io_mfdataset():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET_WILDCARD)
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert len(sonde_ds.attrs['_file_times']) == 7
    assert sonde_ds.attrs['_arm_standards_flag'].OK
    sonde_ds.close()


def test_io_anl_csv():
    headers = ['day', 'month', 'year', 'time', 'pasquill',
               'wdir_60m', 'wspd_60m', 'wdir_60m_std',
               'temp_60m', 'wdir_10m', 'wspd_10m',
               'wdir_10m_std', 'temp_10m', 'temp_dp',
               'rh', 'avg_temp_diff', 'total_precip',
               'solar_rad', 'net_rad', 'atmos_press',
               'wv_pressure', 'temp_soil_10cm',
               'temp_soil_100cm', 'temp_soil_10ft']
    anl_ds = act.io.csvfiles.read_csv(
        act.tests.EXAMPLE_ANL_CSV, sep='\s+', column_names=headers)
    assert 'temp_60m' in anl_ds.variables.keys()
    assert 'rh' in anl_ds.variables.keys()
    assert anl_ds['temp_60m'].values[10] == -1.7
    anl_ds.close()


def test_io_dod():
    dims = {'time': 1440, 'drop_diameter': 50}
    obj = act.io.armfiles.create_obj_from_arm_dod('vdis.b1', dims, version='1.2', scalar_fill_dim='time')

    assert 'moment1' in obj
    assert len(obj['base_time'].values) == 1440
    assert len(obj['drop_diameter'].values) == 50
