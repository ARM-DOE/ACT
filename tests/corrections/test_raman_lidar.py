import numpy as np

import act


def test_correct_rl():
    # Using ceil data in RL place to save memory
    files = act.tests.sample_files.EXAMPLE_RL1
    ds = act.io.arm.read_arm_netcdf(files)

    ds = act.corrections.raman_lidar.correct_rl(ds, range_normalize_log_values=True)
    np.testing.assert_almost_equal(np.max(ds['depolarization_counts_high'].values), 9.91, decimal=2)
    np.testing.assert_almost_equal(
        np.min(ds['depolarization_counts_high'].values), -7.00, decimal=2
    )
    np.testing.assert_almost_equal(
        np.mean(ds['depolarization_counts_high'].values), -1.45, decimal=2
    )
