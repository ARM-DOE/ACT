import numpy as np

from act.io.arm import read_arm_netcdf
from act.qc.radiometer_tests import fft_shading_test
from act.tests import EXAMPLE_MFRSR


def test_fft_shading_test():
    ds = read_arm_netcdf(EXAMPLE_MFRSR)
    ds.clean.cleanup()
    ds = fft_shading_test(ds)
    qc_data = ds['qc_diffuse_hemisp_narrowband_filter4']
    assert np.nansum(qc_data.values) == 7164
