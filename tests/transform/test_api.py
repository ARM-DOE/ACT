"""Tests for the public act.transform API and dataset accessor."""

import numpy as np
import pytest
import xarray as xr
from _helpers import _da

import act


class TestFunctionAccessor:
    def test_returns_arm_qc_metadata(self):
        da = _da([0.0, 1.0, 2.0])
        target = np.array([0.5, 1.5])
        result, qc = act.transform.interpolate(da, target, dim="time")
        assert isinstance(qc, xr.DataArray)
        assert qc.shape == result.shape
        assert qc.attrs.get("standard_name") == "quality_flag"
        assert qc.attrs["flag_masks"] == [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ]
        assert qc.attrs["flag_meanings"] == act.transform.constants.QC_FLAG_MEANINGS
        assert qc.attrs["flag_assessments"] == [
            "Bad",
            "Indeterminate",
            "Indeterminate",
            "Indeterminate",
            "Indeterminate",
            "Indeterminate",
            "Indeterminate",
            "Bad",
            "Bad",
            "Bad",
            "Indeterminate",
            "Bad",
            "Indeterminate",
        ]
        assert qc.attrs["flag_comments"] == act.transform.constants.QC_FLAG_COMMENTS

    def test_invalid_dim_raises(self):
        da = _da([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="not found"):
            act.transform.interpolate(da, np.array([0.5]), dim="height")

    def test_interpolate_accessor_matches_function(self):
        da = _da([0.0, 1.0, 4.0, 9.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
        ds = xr.Dataset({"temp": da})
        target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=["time"])
        expected, expected_qc = act.transform.interpolate(da, target, dim="time")
        result, qc = ds.transform.interpolate("temp", target, dim="time")
        np.testing.assert_allclose(result.values, expected.values)
        np.testing.assert_array_equal(qc.values, expected_qc.values)

    def test_attrs_preserved(self):
        da = _da([1.0, 2.0, 3.0])
        da.attrs["units"] = "K"
        target = np.array([0.5, 1.5])
        result, _ = act.transform.bin_average(da, target, dim="time")
        assert result.attrs.get("units") == "K"

    def test_bin_average_accessor_matches_function(self):
        da = _da([0.0, 2.0, 4.0, 6.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
        ds = xr.Dataset({"temp": da})
        target = xr.DataArray(np.array([1.0, 3.0]), dims=["time"])
        expected, _ = act.transform.bin_average(da, target, dim="time")
        result, _ = ds.transform.bin_average("temp", target, dim="time")
        np.testing.assert_allclose(result.values, expected.values)

    def test_subsample_accessor_matches_function(self):
        da = _da([10.0, 20.0, 30.0], coord=np.array([0.0, 1.0, 2.0]))
        ds = xr.Dataset({"temp": da})
        target = np.array([0.1, 0.9, 1.9])
        expected, _ = act.transform.subsample(da, target, dim="time", t_range=0.5)
        result, _ = ds.transform.subsample("temp", target, dim="time", t_range=0.5)
        np.testing.assert_allclose(result.values, expected.values)


class TestTransformDataset:
    def _make_ds(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        return xr.Dataset(
            {
                "temp": xr.DataArray(
                    [10.0, 20.0, 30.0, 40.0], coords={"time": time}, dims=["time"]
                ),
                "qc_temp": xr.DataArray(
                    [0, 0, 0, 0],
                    coords={"time": time},
                    dims=["time"],
                    attrs={"flag_masks": [1], "flag_assessments": ["Bad"]},
                ),
                "pressure": xr.DataArray(
                    [1000.0, 900.0, 800.0, 700.0], coords={"time": time}, dims=["time"]
                ),
            }
        )

    def test_transforms_all_dim_vars(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=["time"])
        result = act.transform.transform_dataset(ds, target, dim="time", transform="interpolate")
        assert "temp" in result
        assert "pressure" in result
        assert result["temp"].shape == (3,)
        assert result["pressure"].shape == (3,)

    def test_qc_companions_included(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5]), dims=["time"])
        result = act.transform.transform_dataset(ds, target, dim="time", transform="interpolate")
        assert "qc_temp" in result

    def test_dataset_attrs_preserved(self):
        ds = self._make_ds()
        ds.attrs["source"] = "test"
        target = xr.DataArray(np.array([0.5, 1.5]), dims=["time"])
        result = act.transform.transform_dataset(
            ds, target, dim="time", transform="subsample", t_range=1.0
        )
        assert result.attrs.get("source") == "test"

    def test_invalid_transform_raises(self):
        ds = self._make_ds()
        with pytest.raises(ValueError, match="Unknown transform"):
            act.transform.transform_dataset(ds, np.array([0.5]), dim="time", transform="magic")

    def test_per_variable_controls(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5]), dims=["time"])
        result = act.transform.transform_dataset(
            ds,
            target,
            dim="time",
            transform="interpolate",
            per_var_transform={"temp": "subsample"},
            per_var_kwargs={"temp": {"t_range": 2.0}},
        )
        assert "temp" in result
        assert "pressure" in result

    def test_target_ds_shorthand(self):
        ds = self._make_ds()
        target_ds = xr.Dataset(
            {
                "other": xr.DataArray(
                    [1.0, 2.0], coords={"time": np.array([0.5, 1.5])}, dims=["time"]
                )
            }
        )
        result = act.transform.transform_dataset(
            ds, dim="time", transform="interpolate", target_ds=target_ds
        )
        assert result["time"].shape == (2,)
        assert list(result["time"].values) == [0.5, 1.5]

    def test_output_bounds_from_target_dataset_are_preserved(self):
        ds = self._make_ds()
        target_time = np.array([0.5, 1.5])
        target_bounds = xr.DataArray(
            [[0.0, 1.0], [1.0, 2.0]],
            coords={"time": target_time},
            dims=["time", "bound"],
            name="time_bounds",
        )
        target_ds = xr.Dataset({"time_bounds": target_bounds})
        target_ds = target_ds.assign_coords(time=target_time)
        target_ds["time"].attrs["bounds"] = "time_bounds"

        result = act.transform.transform_dataset(
            ds, target_ds=target_ds, dim="time", transform="interpolate"
        )

        assert result["time"].attrs["bounds"] == "time_bounds"
        np.testing.assert_array_equal(result["time_bounds"].values, target_bounds.values)
        assert result["time_bounds"].sizes["time"] == result.sizes["time"]

    def test_output_coordinate_drops_unavailable_bounds_reference(self):
        ds = self._make_ds()
        ds["time"].attrs["bounds"] = "time_bounds"
        target = xr.DataArray(np.array([0.5, 1.5]), dims=["time"])

        result = act.transform.transform_dataset(
            ds, target=target, dim="time", transform="interpolate"
        )

        assert "bounds" not in result["time"].attrs
        assert "time_bounds" not in result

    def test_bounds_autodetect(self):
        time = np.array([1.0, 3.0])
        time_bounds = xr.DataArray(
            [[0.0, 2.0], [2.0, 4.0]], coords={"time": time}, dims=["time", "bound"]
        )
        temp = xr.DataArray([10.0, 20.0], coords={"time": time}, dims=["time"])
        ds = xr.Dataset({"temp": temp})
        ds["time_bounds"] = time_bounds
        ds["time"].attrs["bounds"] = "time_bounds"

        target_time = np.array([2.0])
        target_bounds = xr.DataArray(
            [[0.0, 4.0]], coords={"time": target_time}, dims=["time", "bound"]
        )
        target_ds = xr.Dataset(
            {"other": xr.DataArray([1.0], coords={"time": target_time}, dims=["time"])}
        )
        target_ds["target_bounds"] = target_bounds
        target_ds["time"].attrs["bounds"] = "target_bounds"

        result = act.transform.transform_dataset(
            ds, dim="time", transform="bin_average", target_ds=target_ds
        )
        assert result["temp"].values[0] == pytest.approx(15.0)

    def test_coord_encoding_preservation(self):
        time = np.array([0.0, 1.0, 2.0])
        da = xr.DataArray([1.0, 2.0, 3.0], coords={"time": time}, dims=["time"], name="data")
        da["time"].attrs["units"] = "seconds since 2026-01-01"
        da["time"].encoding["calendar"] = "standard"

        target = xr.DataArray([0.5, 1.5], dims=["time"])
        res, _ = act.transform.interpolate(da, target, dim="time")
        assert res["time"].attrs.get("units") == "seconds since 2026-01-01"
        assert res["time"].encoding.get("calendar") == "standard"

    def test_accessor_matches_function(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5]), dims=["time"])
        expected = act.transform.transform_dataset(ds, target, dim="time", transform="interpolate")
        result = ds.transform.transform_dataset(target=target, dim="time", transform="interpolate")
        np.testing.assert_allclose(result["temp"].values, expected["temp"].values)


class TestMakeCoord:
    def test_make_coord_datetime(self):
        coord = act.transform.make_coord(
            "2026-06-29T12:00:00", "2026-06-29T14:00:00", "1h", name="time"
        )
        assert isinstance(coord, xr.DataArray)
        assert coord.name == "time"
        assert coord.dims == ("time",)
        assert len(coord) == 3
        assert np.issubdtype(coord.dtype, np.datetime64)

    def test_make_coord_numeric(self):
        coord = act.transform.make_coord(0.0, 10.0, 2.5, name="height")
        assert isinstance(coord, xr.DataArray)
        assert coord.name == "height"
        assert coord.dims == ("height",)
        assert list(coord.values) == [0.0, 2.5, 5.0, 7.5]


class TestPublicSurface:
    """The public surface must be discoverable, not merely reachable.

    sphinx.ext.autosummary builds a module's API page from dir(), so a name
    missing from dir() is silently omitted from the generated documentation even
    though getattr() finds it.
    """

    def test_dir_matches_all(self):
        assert sorted(dir(act.transform)) == sorted(act.transform.__all__)

    @pytest.mark.parametrize(
        "name",
        [
            "bin_average",
            "interpolate",
            "subsample",
            "make_coord",
            "transform_dataset",
            "Transform",
        ],
    )
    def test_public_name_is_discoverable(self, name):
        # The three transforms are imported eagerly rather than through
        # lazy.attach, so they are the ones at risk of dropping out of dir().
        assert name in dir(act.transform)
        assert name in act.transform.__all__
        assert getattr(act.transform, name) is not None

    def test_qc_bit_constants_exported(self):
        # Every individual flag bit is re-exported at package level. The
        # QC_ALL_FLAGS / QC_FLAG_MEANINGS / QC_FLAG_ASSESSMENTS lookup tables are
        # deliberately not, and are reached through act.transform.constants.
        for bit in act.transform.constants.QC_ALL_FLAGS:
            names = [
                name
                for name, value in vars(act.transform.constants).items()
                if name.startswith("QC_") and value is bit
            ]
            assert names, f"no constant found for bit {bit}"
            for name in names:
                assert name in act.transform.__all__, name
                assert getattr(act.transform, name) == bit
