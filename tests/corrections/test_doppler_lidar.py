import numpy as np

import act


def test_correct_dl():
    # Test the DL correction script on a PPI dataset eventhough it will
    # mostlikely be used on FPT scans. Doing this to save space with only
    # one datafile in the repo.
    files = act.tests.sample_files.EXAMPLE_DLPPI
    ds = act.io.arm.read_arm_netcdf(files)

    new_ds = act.corrections.doppler_lidar.correct_dl(ds, fill_value=np.nan)
    data = new_ds['attenuated_backscatter'].values
    np.testing.assert_almost_equal(np.nansum(data), -186479.83, decimal=0.1)

    new_ds = act.corrections.doppler_lidar.correct_dl(ds, range_normalize=False)
    data = new_ds['attenuated_backscatter'].values
    np.testing.assert_almost_equal(np.nansum(data), -200886.0, decimal=0.1)
