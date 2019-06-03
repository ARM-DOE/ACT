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
