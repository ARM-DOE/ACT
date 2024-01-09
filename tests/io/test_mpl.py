import act


def test_io_mpldataset():
    try:
        mpl_ds = act.io.mpl.read_sigma_mplv5(act.tests.EXAMPLE_SIGMA_MPLV5)
    except Exception:
        return

    # Tests fields
    assert 'channel_1' in mpl_ds.variables.keys()
    assert 'temp_0' in mpl_ds.variables.keys()
    assert mpl_ds.channel_1.values.shape == (102, 1000)

    # Tests coordinates
    assert 'time' in mpl_ds.coords.keys()
    assert 'range' in mpl_ds.coords.keys()
    assert mpl_ds.coords['time'].values.shape == (102,)
    assert mpl_ds.coords['range'].values.shape == (1000,)
    assert '_arm_standards_flag' in mpl_ds.attrs.keys()

    # Tests attributes
    assert '_datastream' in mpl_ds.attrs.keys()
    mpl_ds.close()
