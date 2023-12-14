import numpy as np

import act


def test_correct_mpl():
    # Make a fake ARM dataset to test with, just an array with 1e-7 for half
    # of it
    test_data = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_MPL_1SAMPLE)
    ds = act.corrections.mpl.correct_mpl(test_data)
    sig_cross_pol = ds['signal_return_cross_pol'].values[1, 10:15]
    sig_co_pol = ds['signal_return_co_pol'].values[1, 10:15]
    height = ds['height'].values[0:10]
    overlap0 = ds['overlap_correction'].values[1, 0, 0:5]
    overlap1 = ds['overlap_correction'].values[1, 1, 0:5]
    overlap2 = ds['overlap_correction'].values[1, 2, 0:5]
    np.testing.assert_allclose(overlap0, [0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(overlap1, [754.338, 754.338, 754.338, 754.338, 754.338])
    np.testing.assert_allclose(overlap2, [181.9355, 181.9355, 181.9355, 181.9355, 181.9355])
    np.testing.assert_allclose(
        sig_cross_pol,
        [-0.5823283, -1.6066532, -1.7153032, -2.520143, -2.275405],
        rtol=4e-06,
    )
    np.testing.assert_allclose(
        sig_co_pol, [12.5631485, 11.035495, 11.999875, 11.09393, 11.388968], rtol=1e-6
    )
    np.testing.assert_allclose(
        height,
        [
            0.00749012,
            0.02247084,
            0.03745109,
            0.05243181,
            0.06741206,
            0.08239277,
            0.09737302,
            0.11235374,
            0.12733398,
            0.14231472,
        ],
        rtol=1e-6,
    )
    assert ds['signal_return_co_pol'].attrs['units'] == '10 * log10(count/us)'
    assert ds['signal_return_cross_pol'].attrs['units'] == '10 * log10(count/us)'
    assert ds['cross_co_ratio'].attrs['long_name'] == 'Cross-pol / Co-pol ratio * 100'
    assert ds['cross_co_ratio'].attrs['units'] == '1'
    assert 'description' not in ds['cross_co_ratio'].attrs.keys()
    assert 'ancillary_variables' not in ds['cross_co_ratio'].attrs.keys()
    assert np.all(np.round(ds['cross_co_ratio'].data[0, 500]) == 34.0)
    assert np.all(np.round(ds['signal_return_co_pol'].data[0, 11]) == 11)
    assert np.all(np.round(ds['signal_return_co_pol'].data[0, 500]) == -6)
    test_data.close()
    ds.close()
