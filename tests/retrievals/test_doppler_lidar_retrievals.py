import numpy as np

import act


def test_doppler_lidar_winds():
    # Process a single file
    dl_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_DLPPI)
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds, intensity_name='intensity')
    assert np.round(np.nansum(result['wind_speed'].values)).astype(int) == 1570
    assert np.round(np.nansum(result['wind_direction'].values)).astype(int) == 32635
    assert result['wind_speed'].attrs['units'] == 'm/s'
    assert result['wind_direction'].attrs['units'] == 'degree'
    assert result['height'].attrs['units'] == 'm'
    dl_ds.close()

    # Process multiple files
    dl_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_DLPPI_MULTI)
    del dl_ds['range'].attrs['units']
    result = act.retrievals.doppler_lidar.compute_winds_from_ppi(dl_ds)
    assert np.round(np.nansum(result['wind_speed'].values)).astype(int) == 2854
    assert np.round(np.nansum(result['wind_direction'].values)).astype(int) == 64986
    dl_ds.close()

def test_doppler_lidar_tucker_retrieval():
    dl_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_DLFPT)
    dl_ds = act.retrievals.calculate_tucker_method_pbl(dl_ds, interval="10min")
    assert np.allclose(dl_ds['pbl_tucker'].values, [675., 795., 735., 825., 795., 735.])
    assert np.allclose(dl_ds['tucker_atmospheric_variance'].values.max(), 19.012731495487955)
    assert np.allclose(dl_ds['tucker_noise_variance'].values.max(), 145.83166344062627)
    dl_ds.close()