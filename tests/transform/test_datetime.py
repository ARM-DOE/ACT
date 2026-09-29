"""Cross-transform datetime coordinates and ARM reader integration."""

import datetime

import cftime
import numpy as np
import pytest
import xarray as xr
from _helpers import MISSING, _da

import act


class TestDatetimeCoordinates:
    """Datetime-like coordinates and bounds must work in every transform.

    ``interpolate`` and ``subsample`` used to raise ``UFuncTypeError`` on a
    real ``datetime64`` coordinate, and the bounds path rejected the
    dtype-object ``cftime`` bounds that ``act.io.arm.read_arm_netcdf``
    produces. Resampling onto a different time base is the headline use case,
    so all three transforms are exercised here.
    """

    @staticmethod
    def _datetime_da(n=120, name="temp"):
        time = np.arange(n, dtype="timedelta64[m]").astype(
            "timedelta64[ns]"
        ) + np.datetime64("2023-01-01T00:00", "ns")
        values = np.arange(n, dtype=float)
        return xr.DataArray(values, coords={"time": time}, dims=["time"], name=name)

    @staticmethod
    def _cftime_bounds(n):
        """Build an (n, 2) dtype-object cftime bounds array of 1-minute cells."""
        base = cftime.DatetimeGregorian(2023, 1, 1, 0, 0)
        edges = [base + datetime.timedelta(minutes=i) for i in range(n + 1)]
        bounds = np.empty((n, 2), dtype=object)
        for i in range(n):
            bounds[i, 0] = edges[i]
            bounds[i, 1] = edges[i + 1]
        return bounds

    @pytest.mark.parametrize("transform", ["bin_average", "interpolate", "subsample"])
    def test_datetime64_coord_runs(self, transform):
        da = self._datetime_da()
        target = act.transform.make_coord(
            "2023-01-01T00:00", "2023-01-01T02:00", "30min"
        )

        result, qc = getattr(act.transform, transform)(da, target, dim="time")

        assert result.shape == target.shape
        assert np.issubdtype(result["time"].dtype, np.datetime64)
        np.testing.assert_array_equal(result["time"].values, target.values)
        # Something real came back -- not an all-missing array.
        assert np.any(result.values != MISSING)
        assert qc.shape == result.shape

    @pytest.mark.parametrize("transform", ["bin_average", "interpolate", "subsample"])
    def test_datetime64_matches_numeric_nanoseconds(self, transform):
        """A datetime axis must give the same numbers as the equivalent numeric axis."""
        da = self._datetime_da()
        target = act.transform.make_coord(
            "2023-01-01T00:00", "2023-01-01T02:00", "30min"
        )

        numeric_da = xr.DataArray(
            da.values,
            coords={
                "time": da["time"].values.astype("datetime64[ns]").astype(np.float64)
            },
            dims=["time"],
            name=da.name,
        )
        numeric_target = target.values.astype("datetime64[ns]").astype(np.float64)

        fn = getattr(act.transform, transform)
        dt_result, dt_qc = fn(da, target, dim="time")
        num_result, num_qc = fn(numeric_da, numeric_target, dim="time")

        np.testing.assert_allclose(dt_result.values, num_result.values)
        np.testing.assert_array_equal(dt_qc.values, num_qc.values)

    @pytest.mark.parametrize("transform", ["bin_average", "interpolate", "subsample"])
    def test_raw_datetime64_target_keeps_datetime_coord(self, transform):
        """A plain numpy datetime64 target (not a DataArray) must not degrade to float."""
        da = self._datetime_da()
        target = np.arange(
            "2023-01-01T00:00",
            "2023-01-01T02:00",
            np.timedelta64(30, "m"),
            dtype="datetime64[ns]",
        )

        result, _ = getattr(act.transform, transform)(da, target, dim="time")

        assert np.issubdtype(result["time"].dtype, np.datetime64)
        np.testing.assert_array_equal(result["time"].values, target)

    def test_explicit_datetime64_bounds(self):
        da = self._datetime_da()
        time = da["time"].values
        bounds = np.stack([time, time + np.timedelta64(1, "m")], axis=1)
        target = act.transform.make_coord(
            "2023-01-01T00:00", "2023-01-01T02:00", "30min"
        )

        result, _ = act.transform.bin_average(
            da, target, dim="time", input_bounds=bounds
        )

        assert np.any(result.values != MISSING)
        # Same bounds expressed in coarser units must agree -- normalization
        # goes through a common unit rather than trusting the raw integers.
        coarse, _ = act.transform.bin_average(
            da, target, dim="time", input_bounds=bounds.astype("datetime64[s]")
        )
        np.testing.assert_allclose(result.values, coarse.values)

    def test_explicit_cftime_bounds(self):
        """dtype-object cftime bounds, as read_arm_netcdf(use_cftime=True) produces them."""
        da = self._datetime_da()
        bounds = self._cftime_bounds(da.sizes["time"])
        assert bounds.dtype == object
        target = act.transform.make_coord(
            "2023-01-01T00:00", "2023-01-01T02:00", "30min"
        )

        result, _ = act.transform.bin_average(
            da, target, dim="time", input_bounds=bounds
        )

        # Must match the identical bounds expressed as datetime64.
        expected, _ = act.transform.bin_average(
            da, target, dim="time", input_bounds=bounds.astype("datetime64[ns]")
        )
        np.testing.assert_allclose(result.values, expected.values)

    def test_transform_dataset_cftime_bounds_autodetect(self):
        """CF bounds auto-detection with a reader-shaped dtype-object cftime bounds array."""
        da = self._datetime_da()
        bounds = self._cftime_bounds(da.sizes["time"])

        ds = xr.Dataset({"temp": da})
        ds["time_bounds"] = xr.DataArray(bounds, dims=["time", "bound"])
        ds["time"].attrs["bounds"] = "time_bounds"

        target = act.transform.make_coord(
            "2023-01-01T00:00", "2023-01-01T02:00", "30min"
        )
        result = act.transform.transform_dataset(
            ds, target=target, dim="time", transform="bin_average"
        )

        assert np.any(result["temp"].values != MISSING)
        # The bounds variable is metadata: consumed as bounds, never transformed.
        assert "time_bounds" not in result
        expected, _ = act.transform.bin_average(
            da, target, dim="time", input_bounds=bounds
        )
        np.testing.assert_allclose(result["temp"].values, expected.values)

    def test_timedelta_t_range(self):
        """A timedelta t_range must be comparable with a datetime coordinate."""
        da = self._datetime_da()
        target = act.transform.make_coord(
            "2023-01-01T00:00", "2023-01-01T02:00", "30min"
        )

        numeric_da = da.assign_coords(time=da["time"].values.astype(np.float64))
        numeric_target = target.values.astype("datetime64[ns]").astype(np.float64)

        result, _ = act.transform.subsample(
            da, target, dim="time", t_range=np.timedelta64(30, "s")
        )
        expected, _ = act.transform.subsample(
            numeric_da, numeric_target, dim="time", t_range=30e9
        )

        np.testing.assert_allclose(result.values, expected.values)

    @pytest.mark.parametrize("transform", ["interpolate", "subsample"])
    @pytest.mark.parametrize(
        "datetime_coord", [True, False], ids=["datetime", "numeric"]
    )
    def test_mismatched_t_range_type_raises(self, transform, datetime_coord):
        if datetime_coord:
            da = self._datetime_da()
            target = act.transform.make_coord(
                "2023-01-01T00:00", "2023-01-01T02:00", "30min"
            )
            t_range, message = 120, "timedelta"
        else:
            da = _da([0.0, 1.0, 2.0])
            target = np.array([0.5])
            t_range, message = np.timedelta64(1, "s"), "number"
        with pytest.raises(TypeError, match=message):
            getattr(act.transform, transform)(da, target, dim="time", t_range=t_range)


class TestDatetimeCoordinatesFromReader:
    """The cftime bounds path as ACT's own ARM reader really produces it."""

    def test_read_arm_netcdf_cftime_bounds(self):
        ds = act.io.arm.read_arm_netcdf([act.tests.EXAMPLE_CEIL1])

        bounds_name = ds["time"].attrs.get("bounds")
        assert bounds_name == "time_bounds"
        # Guard the premise: the reader leaves bounds as cftime objects.
        assert ds[bounds_name].dtype == object
        assert isinstance(ds[bounds_name].values.flat[0], cftime.datetime)

        var = "first_cbh"
        time = ds["time"].values
        target = act.transform.make_coord(time[0], time[-1], "30min")

        # Explicit bounds straight off the reader.
        result, _ = act.transform.bin_average(
            ds[var], target, dim="time", input_bounds=ds[bounds_name].values
        )
        assert np.issubdtype(result["time"].dtype, np.datetime64)

        # And through transform_dataset's CF bounds auto-detection.
        subset = ds[[var, bounds_name]]
        out = act.transform.transform_dataset(
            subset, target=target, dim="time", transform="bin_average"
        )
        np.testing.assert_allclose(out[var].values, result.values)
        ds.close()

    @pytest.mark.parametrize("transform", ["bin_average", "interpolate", "subsample"])
    def test_reader_datetime_axis_all_transforms(self, transform):
        ds = act.io.arm.read_arm_netcdf([act.tests.EXAMPLE_CEIL1])
        time = ds["time"].values
        target = act.transform.make_coord(time[0], time[-1], "30min")

        result, qc = getattr(act.transform, transform)(
            ds["first_cbh"], target, dim="time"
        )

        assert result.shape == target.shape
        assert np.issubdtype(result["time"].dtype, np.datetime64)
        assert qc.shape == result.shape
        ds.close()
