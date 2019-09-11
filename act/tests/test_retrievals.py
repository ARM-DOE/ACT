import act
import numpy as np


def test_get_stability_indices():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_SONDE1)

    sonde_ds = act.retrievals.calculate_stability_indicies(
        sonde_ds, temp_name="tdry", td_name="dp", p_name="pres")
    assert sonde_ds["lifted_index"].units == "kelvin"
    np.testing.assert_almost_equal(sonde_ds["lifted_index"], 28.4639, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["most_unstable_cin"], 5.277, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["surface_based_cin"], 5.277, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["most_unstable_cape"], 1.628, decimal=3)
    np.testing.assert_almost_equal(sonde_ds["surface_based_cape"], 1.628, decimal=3)
    np.testing.assert_almost_equal(
        sonde_ds["lifted_condensation_level_pressure"], 927.143, decimal=3)
    np.testing.assert_almost_equal(
        sonde_ds["lifted_condensation_level_temperature"], -8.079, decimal=3)
    sonde_ds.close()


def test_generic_sobel_cbh():
    ceil = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_CEIL1)

    ceil = ceil.resample(time='1min').nearest()
    ceil = act.retrievals.cbh.generic_sobel_cbh(ceil, variable='backscatter',
                                                height_dim='range', var_thresh=1000.,
                                                fill_na=0)
    cbh = ceil['cbh_sobel'].values
    assert cbh[500] == 615.
    assert cbh[1000] == 555.
    ceil.close()


def test_calculate_precipitable_water():
    sonde_ds = act.io.armfiles.read_netcdf(
        act.tests.sample_files.EXAMPLE_SONDE1)
    assert sonde_ds["tdry"].units == "C", "Temperature must be in Celsius"
    assert sonde_ds["rh"].units == "%", "Relative Humidity must be a percentage"
    assert sonde_ds["pres"].units == "hPa", "Pressure must be in hPa"
    pwv_data = act.retrievals.pwv_calc.calculate_precipitable_water(
        sonde_ds, temp_name='tdry', rh_name='rh', pres_name='pres')
    np.testing.assert_almost_equal(pwv_data, 0.8028, decimal=3)
    sonde_ds.close()
