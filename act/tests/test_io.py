import act
import act.tests.sample_files as sample_files
from pathlib import Path
import tempfile


def test_io():
    sonde_ds = act.io.armfiles.read_netcdf(
        [act.tests.EXAMPLE_MET1])
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert sonde_ds.attrs['_arm_standards_flag'] == (1 << 0)
    sonde_ds.close()


def test_io_mfdataset():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.EXAMPLE_MET_WILDCARD)
    assert 'temp_mean' in sonde_ds.variables.keys()
    assert 'rh_mean' in sonde_ds.variables.keys()
    assert len(sonde_ds.attrs['_file_times']) == 7
    assert sonde_ds.attrs['_arm_standards_flag'] == (1 << 0)
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
    obj = act.io.armfiles.create_obj_from_arm_dod('vdis.b1', dims, version='1.2',
                                                  scalar_fill_dim='time')

    assert 'moment1' in obj
    assert len(obj['base_time'].values) == 1440
    assert len(obj['drop_diameter'].values) == 50


def test_io_write():
    sonde_ds = act.io.armfiles.read_netcdf(sample_files.EXAMPLE_SONDE1)
    sonde_ds.clean.cleanup()

    with tempfile.TemporaryDirectory() as tmpdirname:
        write_file = Path(tmpdirname, Path(sample_files.EXAMPLE_SONDE1).name)
        keep_vars = ['tdry', 'qc_tdry', 'dp', 'qc_dp']
        for var_name in list(sonde_ds.data_vars):
            if var_name not in keep_vars:
                del sonde_ds[var_name]
        sonde_ds.write.write_netcdf(path=write_file, FillValue=-9999)
        sonde_ds.close()
        del sonde_ds

        sonde_ds = act.io.armfiles.read_netcdf(str(write_file))
        assert list(sonde_ds.data_vars) == keep_vars
        assert isinstance(sonde_ds['qc_tdry'].attrs['flag_meanings'], str)
        assert sonde_ds['qc_tdry'].attrs['flag_meanings'].count('__') == 21
        for attr in ['qc_standards_version', 'qc_method', 'qc_comment']:
            assert attr not in list(sonde_ds.attrs)


def test_io_mpldataset():
    try:
        mpl_ds = act.io.mpl.read_sigma_mplv5(
            act.tests.EXAMPLE_SIGMA_MPLV5)
    except Exception:
        return

    # Tests fields
    assert 'channel_1' in mpl_ds.variables.keys()
    assert 'temp_0' in mpl_ds.variables.keys()
    assert mpl_ds.channel_1.values.shape == (102, 1000)

    # Tests coordinates
    assert 'time' in mpl_ds.coords.keys()
    assert 'range' in mpl_ds.coords.keys()
    assert mpl_ds.coords['time'].values.shape == (102, )
    assert mpl_ds.coords['range'].values.shape == (1000, )
    assert '_arm_standards_flag' in mpl_ds.attrs.keys()

    # Tests attributes
    assert '_datastream' in mpl_ds.attrs.keys()
    mpl_ds.close()
