"""Tests for the act.transform function API and ds.transform accessor."""

import datetime

import cftime
import numpy as np
import pytest
import xarray as xr

import act

MISSING = -9999.0


def _da(values, coord_name='time', coord=None, name='temp'):
    if coord is None:
        coord = np.arange(len(values), dtype=float)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        coords={coord_name: coord},
        dims=[coord_name],
        name=name,
    )


class TestInterpolate:
    def test_basic_1d(self):
        da = _da([0.0, 1.0, 4.0, 9.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
        target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=['time'])
        result, qc = act.transform.interpolate(da, target, dim='time')
        assert result.shape == (3,)
        assert result.values[0] == pytest.approx(0.5)
        assert result.values[1] == pytest.approx(2.5)
        assert result.name == 'temp'

    def test_returns_arm_qc_metadata(self):
        da = _da([0.0, 1.0, 2.0])
        target = np.array([0.5, 1.5])
        result, qc = act.transform.interpolate(da, target, dim='time')
        assert isinstance(qc, xr.DataArray)
        assert qc.shape == result.shape
        assert qc.attrs.get('standard_name') == 'quality_flag'
        assert qc.attrs['flag_masks'] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        assert qc.attrs['flag_meanings'] == act.transform.constants.QC_FLAG_MEANINGS
        assert qc.attrs['flag_assessments'] == [
            'Bad',
            'Indeterminate',
            'Indeterminate',
            'Indeterminate',
            'Indeterminate',
            'Indeterminate',
            'Indeterminate',
            'Bad',
            'Bad',
            'Bad',
            'Indeterminate',
            'Bad',
            'Indeterminate',
        ]
        assert qc.attrs['flag_comments'] == act.transform.constants.QC_FLAG_COMMENTS

    def test_missing_value_respected(self):
        da = _da([0.0, MISSING, 4.0], coord=np.array([0.0, 1.0, 2.0]))
        da.encoding['_FillValue'] = MISSING
        target = np.array([0.5, 1.5])
        result, qc = act.transform.interpolate(da, target, dim='time')
        assert qc.values[1] != 0

    def test_invalid_dim_raises(self):
        da = _da([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match='not found'):
            act.transform.interpolate(da, np.array([0.5]), dim='height')

    def test_2d_dataarray(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        height = np.array([100.0, 200.0, 300.0])
        data = np.arange(12, dtype=float).reshape(4, 3)
        da = xr.DataArray(
            data,
            coords={'time': time, 'height': height},
            dims=['time', 'height'],
            name='wind',
        )
        target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=['time'])
        result, qc = act.transform.interpolate(da, target, dim='time')
        assert result.shape == (3, 3)

    def test_accessor_matches_function(self):
        da = _da([0.0, 1.0, 4.0, 9.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
        ds = xr.Dataset({'temp': da})
        target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=['time'])
        expected, expected_qc = act.transform.interpolate(da, target, dim='time')
        result, qc = ds.transform.interpolate('temp', target, dim='time')
        np.testing.assert_allclose(result.values, expected.values)
        np.testing.assert_array_equal(qc.values, expected_qc.values)


class TestBinAverage:
    def test_basic_1d(self):
        da = _da([0.0, 2.0, 4.0, 6.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
        target = xr.DataArray(np.array([1.0, 3.0]), dims=['time'])
        result, qc = act.transform.bin_average(da, target, dim='time')
        assert result.shape == (2,)

    def test_attrs_preserved(self):
        da = _da([1.0, 2.0, 3.0])
        da.attrs['units'] = 'K'
        target = np.array([0.5, 1.5])
        result, _ = act.transform.bin_average(da, target, dim='time')
        assert result.attrs.get('units') == 'K'

    def test_accessor_matches_function(self):
        da = _da([0.0, 2.0, 4.0, 6.0], coord=np.array([0.0, 1.0, 2.0, 3.0]))
        ds = xr.Dataset({'temp': da})
        target = xr.DataArray(np.array([1.0, 3.0]), dims=['time'])
        expected, _ = act.transform.bin_average(da, target, dim='time')
        result, _ = ds.transform.bin_average('temp', target, dim='time')
        np.testing.assert_allclose(result.values, expected.values)

    def test_qc_mask_assessment_and_default(self):
        da = _da([1.0, 2.0], coord=np.array([0.0, 1.0]))
        qc = xr.DataArray([1, 2], coords=da.coords, dims=da.dims)
        qc.attrs['flag_masks'] = [1, 2]
        qc.attrs['flag_assessments'] = ['Bad', 'Indeterminate']
        target = xr.DataArray([0.0], dims=['time'])
        bounds = np.array([[0.0, 2.0]])
        input_bounds = np.array([[0.0, 1.0], [1.0, 2.0]])

        _, default_qc = act.transform.bin_average(
            da, target, dim='time', qc=qc, input_bounds=input_bounds, output_bounds=bounds
        )
        _, bad_qc = act.transform.bin_average(
            da,
            target,
            dim='time',
            qc=qc,
            qc_mask='Bad',
            input_bounds=input_bounds,
            output_bounds=bounds,
        )
        _, indeterminate_qc = act.transform.bin_average(
            da,
            target,
            dim='time',
            qc=qc,
            qc_mask='Indeterminate',
            input_bounds=input_bounds,
            output_bounds=bounds,
        )

        np.testing.assert_array_equal(default_qc.values, bad_qc.values)
        assert bad_qc.values[0] & act.transform.QC_SOME_BAD_INPUTS
        assert indeterminate_qc.values[0] & act.transform.QC_SOME_BAD_INPUTS

    def test_unknown_qc_assessment_raises(self):
        da = _da([1.0, 2.0])
        qc = xr.DataArray([0, 0], coords=da.coords, dims=da.dims)
        qc.attrs['flag_masks'] = [1]
        qc.attrs['flag_assessments'] = ['Bad']
        with pytest.raises(ValueError, match='not found'):
            act.transform.bin_average(da, [0.5], dim='time', qc=qc, qc_mask='Suspect')

    def test_multiple_qc_assessments_are_combined(self):
        da = _da([1.0, 2.0, 3.0], coord=np.array([0.0, 1.0, 2.0]))
        qc = xr.DataArray([1, 2, 4], coords=da.coords, dims=da.dims)
        qc.attrs['flag_masks'] = [1, 2, 4]
        qc.attrs['flag_assessments'] = ['Bad', 'Suspect', 'Indeterminate']
        target = xr.DataArray([1.0], dims=['time'])
        result, _ = act.transform.bin_average(
            da,
            target,
            dim='time',
            qc=qc,
            qc_mask=['Bad', 'Suspect'],
            input_bounds=np.array([[-0.5, 0.5], [0.5, 1.5], [1.5, 2.5]]),
            output_bounds=np.array([[0.0, 2.0]]),
        )
        assert result.values[0] == pytest.approx(3.0)

    def test_invalid_qc_assessment_list_raises(self):
        da = _da([1.0, 2.0])
        qc = xr.DataArray([0, 0], coords=da.coords, dims=da.dims)
        qc.attrs['flag_masks'] = [1]
        qc.attrs['flag_assessments'] = ['Bad']
        with pytest.raises(TypeError, match='assessment string'):
            act.transform.bin_average(da, [0.5], dim='time', qc=qc, qc_mask=['Bad', 1])

    def test_some_bad_inputs_excludes_all_bad_flag(self):
        da = _da([1.0, 2.0], coord=np.array([0.0, 1.0]))
        qc = xr.DataArray([act.transform.QC_BAD, 0], coords=da.coords, dims=da.dims)
        target = xr.DataArray([0.0], dims=['time'])
        bounds = np.array([[0.0, 2.0]])

        _, result_qc = act.transform.bin_average(
            da,
            target,
            dim='time',
            qc=qc,
            qc_mask=act.transform.QC_BAD,
            input_bounds=np.array([[0.0, 1.0], [1.0, 2.0]]),
            output_bounds=bounds,
        )

        assert result_qc.values[0] & act.transform.QC_SOME_BAD_INPUTS
        assert not result_qc.values[0] & act.transform.QC_ALL_BAD_INPUTS

    def test_all_bad_inputs_excludes_some_bad_flag(self):
        da = _da([1.0, 2.0], coord=np.array([0.0, 1.0]))
        qc = xr.DataArray(
            [act.transform.QC_BAD, act.transform.QC_BAD],
            coords=da.coords,
            dims=da.dims,
        )
        target = xr.DataArray([0.0], dims=['time'])
        bounds = np.array([[0.0, 2.0]])

        _, result_qc = act.transform.bin_average(
            da,
            target,
            dim='time',
            qc=qc,
            qc_mask=act.transform.QC_BAD,
            input_bounds=np.array([[0.0, 1.0], [1.0, 2.0]]),
            output_bounds=bounds,
        )

        assert result_qc.values[0] & act.transform.QC_ALL_BAD_INPUTS
        assert not result_qc.values[0] & act.transform.QC_SOME_BAD_INPUTS


class TestSubsample:
    def test_basic_1d(self):
        da = _da([10.0, 20.0, 30.0], coord=np.array([0.0, 1.0, 2.0]))
        target = np.array([0.1, 0.9, 1.9])
        result, qc = act.transform.subsample(da, target, dim='time', t_range=0.5)
        assert result.shape == (3,)
        assert result.values[0] == pytest.approx(10.0)
        assert result.values[1] == pytest.approx(20.0)
        assert result.values[2] == pytest.approx(30.0)

    def test_accessor_matches_function(self):
        da = _da([10.0, 20.0, 30.0], coord=np.array([0.0, 1.0, 2.0]))
        ds = xr.Dataset({'temp': da})
        target = np.array([0.1, 0.9, 1.9])
        expected, _ = act.transform.subsample(da, target, dim='time', t_range=0.5)
        result, _ = ds.transform.subsample('temp', target, dim='time', t_range=0.5)
        np.testing.assert_allclose(result.values, expected.values)


class TestTransformDataset:
    def _make_ds(self):
        time = np.array([0.0, 1.0, 2.0, 3.0])
        return xr.Dataset(
            {
                'temp': xr.DataArray(
                    [10.0, 20.0, 30.0, 40.0], coords={'time': time}, dims=['time']
                ),
                'qc_temp': xr.DataArray([0, 0, 0, 0], coords={'time': time}, dims=['time']),
                'pressure': xr.DataArray(
                    [1000.0, 900.0, 800.0, 700.0], coords={'time': time}, dims=['time']
                ),
            }
        )

    def test_transforms_all_dim_vars(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5, 2.5]), dims=['time'])
        result = act.transform.transform_dataset(ds, target, dim='time', transform='interpolate')
        assert 'temp' in result
        assert 'pressure' in result
        assert result['temp'].shape == (3,)
        assert result['pressure'].shape == (3,)

    def test_qc_companions_included(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5]), dims=['time'])
        result = act.transform.transform_dataset(ds, target, dim='time', transform='interpolate')
        assert 'qc_temp' in result

    def test_dataset_attrs_preserved(self):
        ds = self._make_ds()
        ds.attrs['source'] = 'test'
        target = xr.DataArray(np.array([0.5, 1.5]), dims=['time'])
        result = act.transform.transform_dataset(
            ds, target, dim='time', transform='subsample', t_range=1.0
        )
        assert result.attrs.get('source') == 'test'

    def test_invalid_transform_raises(self):
        ds = self._make_ds()
        with pytest.raises(ValueError, match='Unknown transform'):
            act.transform.transform_dataset(ds, np.array([0.5]), dim='time', transform='magic')

    def test_per_variable_controls(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5]), dims=['time'])
        result = act.transform.transform_dataset(
            ds,
            target,
            dim='time',
            transform='interpolate',
            per_var_transform={'temp': 'subsample'},
            per_var_kwargs={'temp': {'t_range': 2.0}},
        )
        assert 'temp' in result
        assert 'pressure' in result

    def test_target_ds_shorthand(self):
        ds = self._make_ds()
        target_ds = xr.Dataset(
            {
                'other': xr.DataArray(
                    [1.0, 2.0], coords={'time': np.array([0.5, 1.5])}, dims=['time']
                )
            }
        )
        result = act.transform.transform_dataset(
            ds, dim='time', transform='interpolate', target_ds=target_ds
        )
        assert result['time'].shape == (2,)
        assert list(result['time'].values) == [0.5, 1.5]

    def test_bounds_autodetect(self):
        time = np.array([1.0, 3.0])
        time_bounds = xr.DataArray(
            [[0.0, 2.0], [2.0, 4.0]], coords={'time': time}, dims=['time', 'bound']
        )
        temp = xr.DataArray([10.0, 20.0], coords={'time': time}, dims=['time'])
        ds = xr.Dataset({'temp': temp})
        ds['time_bounds'] = time_bounds
        ds['time'].attrs['bounds'] = 'time_bounds'

        target_time = np.array([2.0])
        target_bounds = xr.DataArray(
            [[0.0, 4.0]], coords={'time': target_time}, dims=['time', 'bound']
        )
        target_ds = xr.Dataset(
            {'other': xr.DataArray([1.0], coords={'time': target_time}, dims=['time'])}
        )
        target_ds['target_bounds'] = target_bounds
        target_ds['time'].attrs['bounds'] = 'target_bounds'

        result = act.transform.transform_dataset(
            ds, dim='time', transform='bin_average', target_ds=target_ds
        )
        assert result['temp'].values[0] == pytest.approx(15.0)

    def test_coord_encoding_preservation(self):
        time = np.array([0.0, 1.0, 2.0])
        da = xr.DataArray([1.0, 2.0, 3.0], coords={'time': time}, dims=['time'], name='data')
        da['time'].attrs['units'] = 'seconds since 2026-01-01'
        da['time'].encoding['calendar'] = 'standard'

        target = xr.DataArray([0.5, 1.5], dims=['time'])
        res, _ = act.transform.interpolate(da, target, dim='time')
        assert res['time'].attrs.get('units') == 'seconds since 2026-01-01'
        assert res['time'].encoding.get('calendar') == 'standard'

    def test_accessor_matches_function(self):
        ds = self._make_ds()
        target = xr.DataArray(np.array([0.5, 1.5]), dims=['time'])
        expected = act.transform.transform_dataset(ds, target, dim='time', transform='interpolate')
        result = ds.transform.transform_dataset(target=target, dim='time', transform='interpolate')
        np.testing.assert_allclose(result['temp'].values, expected['temp'].values)


class TestMakeCoord:
    def test_make_coord_datetime(self):
        coord = act.transform.make_coord(
            '2026-06-29T12:00:00', '2026-06-29T14:00:00', '1h', name='time'
        )
        assert isinstance(coord, xr.DataArray)
        assert coord.name == 'time'
        assert coord.dims == ('time',)
        assert len(coord) == 3
        assert np.issubdtype(coord.dtype, np.datetime64)

    def test_make_coord_numeric(self):
        coord = act.transform.make_coord(0.0, 10.0, 2.5, name='height')
        assert isinstance(coord, xr.DataArray)
        assert coord.name == 'height'
        assert coord.dims == ('height',)
        assert list(coord.values) == [0.0, 2.5, 5.0, 7.5]


class TestDatetimeCoordinates:
    """Datetime-like coordinates and bounds must work in every transform.

    ``interpolate`` and ``subsample`` used to raise ``UFuncTypeError`` on a
    real ``datetime64`` coordinate, and the bounds path rejected the
    dtype-object ``cftime`` bounds that ``act.io.arm.read_arm_netcdf``
    produces. Resampling onto a different time base is the headline use case,
    so all three transforms are exercised here.
    """

    @staticmethod
    def _datetime_da(n=120, name='temp'):
        time = np.arange(n, dtype='timedelta64[m]').astype('timedelta64[ns]') + np.datetime64(
            '2023-01-01T00:00', 'ns'
        )
        values = np.arange(n, dtype=float)
        return xr.DataArray(values, coords={'time': time}, dims=['time'], name=name)

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

    @pytest.mark.parametrize('transform', ['bin_average', 'interpolate', 'subsample'])
    def test_datetime64_coord_runs(self, transform):
        da = self._datetime_da()
        target = act.transform.make_coord('2023-01-01T00:00', '2023-01-01T02:00', '30min')

        result, qc = getattr(act.transform, transform)(da, target, dim='time')

        assert result.shape == target.shape
        assert np.issubdtype(result['time'].dtype, np.datetime64)
        np.testing.assert_array_equal(result['time'].values, target.values)
        # Something real came back -- not an all-missing array.
        assert np.any(result.values != MISSING)
        assert qc.shape == result.shape

    @pytest.mark.parametrize('transform', ['bin_average', 'interpolate', 'subsample'])
    def test_datetime64_matches_numeric_nanoseconds(self, transform):
        """A datetime axis must give the same numbers as the equivalent numeric axis."""
        da = self._datetime_da()
        target = act.transform.make_coord('2023-01-01T00:00', '2023-01-01T02:00', '30min')

        numeric_da = xr.DataArray(
            da.values,
            coords={'time': da['time'].values.astype('datetime64[ns]').astype(np.float64)},
            dims=['time'],
            name=da.name,
        )
        numeric_target = target.values.astype('datetime64[ns]').astype(np.float64)

        fn = getattr(act.transform, transform)
        dt_result, dt_qc = fn(da, target, dim='time')
        num_result, num_qc = fn(numeric_da, numeric_target, dim='time')

        np.testing.assert_allclose(dt_result.values, num_result.values)
        np.testing.assert_array_equal(dt_qc.values, num_qc.values)

    @pytest.mark.parametrize('transform', ['bin_average', 'interpolate', 'subsample'])
    def test_raw_datetime64_target_keeps_datetime_coord(self, transform):
        """A plain numpy datetime64 target (not a DataArray) must not degrade to float."""
        da = self._datetime_da()
        target = np.arange(
            '2023-01-01T00:00', '2023-01-01T02:00', np.timedelta64(30, 'm'), dtype='datetime64[ns]'
        )

        result, _ = getattr(act.transform, transform)(da, target, dim='time')

        assert np.issubdtype(result['time'].dtype, np.datetime64)
        np.testing.assert_array_equal(result['time'].values, target)

    def test_explicit_datetime64_bounds(self):
        da = self._datetime_da()
        time = da['time'].values
        bounds = np.stack([time, time + np.timedelta64(1, 'm')], axis=1)
        target = act.transform.make_coord('2023-01-01T00:00', '2023-01-01T02:00', '30min')

        result, _ = act.transform.bin_average(da, target, dim='time', input_bounds=bounds)

        assert np.any(result.values != MISSING)
        # Same bounds expressed in coarser units must agree -- normalization
        # goes through a common unit rather than trusting the raw integers.
        coarse, _ = act.transform.bin_average(
            da, target, dim='time', input_bounds=bounds.astype('datetime64[s]')
        )
        np.testing.assert_allclose(result.values, coarse.values)

    def test_explicit_cftime_bounds(self):
        """dtype-object cftime bounds, as read_arm_netcdf(use_cftime=True) produces them."""
        da = self._datetime_da()
        bounds = self._cftime_bounds(da.sizes['time'])
        assert bounds.dtype == object
        target = act.transform.make_coord('2023-01-01T00:00', '2023-01-01T02:00', '30min')

        result, _ = act.transform.bin_average(da, target, dim='time', input_bounds=bounds)

        # Must match the identical bounds expressed as datetime64.
        expected, _ = act.transform.bin_average(
            da, target, dim='time', input_bounds=bounds.astype('datetime64[ns]')
        )
        np.testing.assert_allclose(result.values, expected.values)

    def test_transform_dataset_cftime_bounds_autodetect(self):
        """CF bounds auto-detection with a reader-shaped dtype-object cftime bounds array."""
        da = self._datetime_da()
        bounds = self._cftime_bounds(da.sizes['time'])

        ds = xr.Dataset({'temp': da})
        ds['time_bounds'] = xr.DataArray(bounds, dims=['time', 'bound'])
        ds['time'].attrs['bounds'] = 'time_bounds'

        target = act.transform.make_coord('2023-01-01T00:00', '2023-01-01T02:00', '30min')
        result = act.transform.transform_dataset(
            ds, target=target, dim='time', transform='bin_average'
        )

        assert np.any(result['temp'].values != MISSING)
        # The bounds variable is metadata: consumed as bounds, never transformed.
        assert 'time_bounds' not in result
        expected, _ = act.transform.bin_average(da, target, dim='time', input_bounds=bounds)
        np.testing.assert_allclose(result['temp'].values, expected.values)

    def test_timedelta_t_range(self):
        """A timedelta t_range must be comparable with a datetime coordinate."""
        da = self._datetime_da()
        target = act.transform.make_coord('2023-01-01T00:00', '2023-01-01T02:00', '30min')

        result, _ = act.transform.subsample(da, target, dim='time', t_range=np.timedelta64(30, 's'))
        expected, _ = act.transform.subsample(da, target, dim='time', t_range=30e9)

        np.testing.assert_allclose(result.values, expected.values)


class TestDatetimeCoordinatesFromReader:
    """The cftime bounds path as ACT's own ARM reader really produces it."""

    def test_read_arm_netcdf_cftime_bounds(self):
        ds = act.io.arm.read_arm_netcdf([act.tests.EXAMPLE_CEIL1])

        bounds_name = ds['time'].attrs.get('bounds')
        assert bounds_name == 'time_bounds'
        # Guard the premise: the reader leaves bounds as cftime objects.
        assert ds[bounds_name].dtype == object
        assert isinstance(ds[bounds_name].values.flat[0], cftime.datetime)

        var = 'first_cbh'
        time = ds['time'].values
        target = act.transform.make_coord(time[0], time[-1], '30min')

        # Explicit bounds straight off the reader.
        result, _ = act.transform.bin_average(
            ds[var], target, dim='time', input_bounds=ds[bounds_name].values
        )
        assert np.issubdtype(result['time'].dtype, np.datetime64)

        # And through transform_dataset's CF bounds auto-detection.
        subset = ds[[var, bounds_name]]
        out = act.transform.transform_dataset(
            subset, target=target, dim='time', transform='bin_average'
        )
        np.testing.assert_allclose(out[var].values, result.values)
        ds.close()

    @pytest.mark.parametrize('transform', ['bin_average', 'interpolate', 'subsample'])
    def test_reader_datetime_axis_all_transforms(self, transform):
        ds = act.io.arm.read_arm_netcdf([act.tests.EXAMPLE_CEIL1])
        time = ds['time'].values
        target = act.transform.make_coord(time[0], time[-1], '30min')

        result, qc = getattr(act.transform, transform)(ds['first_cbh'], target, dim='time')

        assert result.shape == target.shape
        assert np.issubdtype(result['time'].dtype, np.datetime64)
        assert qc.shape == result.shape
        ds.close()


class TestPublicSurface:
    """The public surface must be discoverable, not merely reachable.

    sphinx.ext.autosummary builds a module's API page from dir(), so a name
    missing from dir() is silently omitted from the generated documentation even
    though getattr() finds it.
    """

    def test_dir_matches_all(self):
        assert sorted(dir(act.transform)) == sorted(act.transform.__all__)

    @pytest.mark.parametrize(
        'name',
        [
            'bin_average',
            'interpolate',
            'subsample',
            'make_coord',
            'transform_dataset',
            'Transform',
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
                if name.startswith('QC_') and value is bit
            ]
            assert names, f'no constant found for bit {bit}'
            for name in names:
                assert name in act.transform.__all__, name
                assert getattr(act.transform, name) == bit
