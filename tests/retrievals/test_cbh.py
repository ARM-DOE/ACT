import act


def test_generic_sobel_cbh():
    ceil = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_CEIL1)

    ceil = ceil.resample(time='1min').nearest()
    ceil = act.retrievals.cbh.generic_sobel_cbh(
        ceil,
        variable='backscatter',
        height_dim='range',
        var_thresh=1000.0,
        fill_na=0.0,
        edge_thresh=5,
    )
    cbh = ceil['cbh_sobel_backscatter'].values
    assert cbh[500] == 615.0
    assert cbh[1000] == 555.0
    ceil.close()
