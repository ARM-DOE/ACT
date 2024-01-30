import glob

import act


def test_io_csv():
    headers = [
        'day',
        'month',
        'year',
        'time',
        'pasquill',
        'wdir_60m',
        'wspd_60m',
        'wdir_60m_std',
        'temp_60m',
        'wdir_10m',
        'wspd_10m',
        'wdir_10m_std',
        'temp_10m',
        'temp_dp',
        'rh',
        'avg_temp_diff',
        'total_precip',
        'solar_rad',
        'net_rad',
        'atmos_press',
        'wv_pressure',
        'temp_soil_10cm',
        'temp_soil_100cm',
        'temp_soil_10ft',
    ]
    anl_ds = act.io.text.read_csv(act.tests.EXAMPLE_ANL_CSV, sep=r'\s+', column_names=headers)
    assert 'temp_60m' in anl_ds.variables.keys()
    assert 'rh' in anl_ds.variables.keys()
    assert anl_ds['temp_60m'].values[10] == -1.7
    anl_ds.close()

    files = glob.glob(act.tests.EXAMPLE_MET_CSV)
    ds = act.io.text.read_csv(files[0])
    assert 'date_time' in ds
    assert '_datastream' in ds.attrs
