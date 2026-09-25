"""
N-D axis-iteration driver for the act.transform kernels.

Applies a 1D transformation kernel (bin_average, interpolate, or subsample)
along a single axis of an N-dimensional array by iterating over all slices
perpendicular to that axis.

"""

import numpy as np
import pandas as pd
import xarray as xr

from act.transform.bin_average import _bin_average_1d, bin_average
from act.transform.constants import add_arm_qc_attrs
from act.transform.interpolate import _bilinear_interpolate_1d, interpolate
from act.transform.subsample import _subsample_1d, subsample

TRANSFORM_NAMES = ('bin_average', 'interpolate', 'subsample')

DEFAULT_MISSING = -9999.0


def _resolve_qc_mask(qc, qc_mask):
    """Resolve an integer QC mask from bitmasks or flag assessments.

    ``None`` selects the ``Bad`` assessment when QC metadata is available and
    otherwise means that no QC bits are excluded. Integer masks are returned
    unchanged. A string or list of strings is matched against the QC variable's
    ``flag_assessments`` metadata and the corresponding ``flag_masks`` are ORed
    together.
    """
    if qc_mask is None:
        assessments = ['Bad']
    elif isinstance(qc_mask, str):
        assessments = [qc_mask]
    elif isinstance(qc_mask, (list, tuple)):
        if not qc_mask or not all(isinstance(item, str) for item in qc_mask):
            raise TypeError("qc_mask lists must contain at least one assessment string")
        assessments = list(qc_mask)
    elif isinstance(qc_mask, (int, np.integer)) and not isinstance(qc_mask, bool):
        return int(qc_mask)
    else:
        raise TypeError(
            "qc_mask must be an integer, assessment string, list of assessment strings, or None"
        )

    if qc is None:
        if qc_mask is None:
            return 0
        raise ValueError(f"Cannot resolve QC assessment(s) {assessments!r} without a QC DataArray")

    # A QC array without CF assessment metadata cannot resolve the implicit
    # default, so retain the historical no-mask behavior for qc_mask=None.
    if qc_mask is None and 'flag_masks' not in qc.attrs:
        return 0

    try:
        flag_masks = qc.attrs['flag_masks']
        flag_assessments = qc.attrs['flag_assessments']
    except KeyError as exc:
        raise ValueError(
            "QC assessment masks require 'flag_masks' and 'flag_assessments' metadata"
        ) from exc

    if len(flag_masks) != len(flag_assessments):
        raise ValueError("QC flag_masks and flag_assessments must have the same length")

    mask = 0
    matched = set()
    for flag_mask, flag_assessment in zip(flag_masks, flag_assessments):
        assessment = str(flag_assessment)
        if assessment in assessments:
            mask |= int(flag_mask)
            matched.add(assessment)

    missing = [assessment for assessment in assessments if assessment not in matched]
    if missing:
        raise ValueError(f"QC assessment(s) {missing!r} were not found in flag_assessments")
    return mask


def _infer_t_range(index):
    """Return the median absolute spacing between index values."""
    if len(index) < 2:
        return np.inf
    return float(np.median(np.abs(np.diff(index))))


def _is_datetime_like_array(arr):
    """Return True if ``arr`` holds datetime64, timedelta64, or date-like objects."""
    if arr.dtype.kind in 'Mm':
        return True
    if arr.dtype == object and arr.size:
        val = arr.reshape(-1)[0]
        return all(hasattr(val, a) for a in ('year', 'month', 'day'))
    return False


def _to_numeric(values):
    """Convert coordinate-like values to float64 for the numeric kernels.

    The kernels do plain arithmetic on their coordinate arguments, so
    datetime-like coordinates have to be reduced to numbers first.
    ``datetime64``/``timedelta64`` arrays and object arrays of date-like
    values (e.g. the :class:`cftime.DatetimeGregorian` objects that
    :func:`act.io.arm.read_arm_netcdf` produces for CF ``bounds`` variables)
    are all normalized to nanoseconds, so values originating from different
    units remain mutually comparable. Numeric input is passed through
    unchanged apart from the float64 cast.

    Parameters
    ----------
    values : array_like
        Coordinate, bounds, or target values of any supported dtype.

    Returns
    -------
    numpy.ndarray
        float64 array; nanoseconds since the epoch for datetime-like input.

    """
    arr = np.asarray(values)

    if arr.dtype.kind == 'M':
        return arr.astype('datetime64[ns]').astype(np.float64)
    if arr.dtype.kind == 'm':
        return arr.astype('timedelta64[ns]').astype(np.float64)

    if _is_datetime_like_array(arr):
        # cftime / datetime.datetime objects: let numpy parse them via
        # datetime64. Fall through to the plain float cast if it can't.
        try:
            return arr.astype('datetime64[ns]').astype(np.float64)
        except (TypeError, ValueError):
            pass

    return arr.astype(np.float64)


def _to_numeric_scalar(value):
    """Convert a scalar distance like ``t_range`` to float64 nanoseconds if it is a duration."""
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype.kind == 'm' or isinstance(value, pd.Timedelta):
        return float(arr.astype('timedelta64[ns]').astype(np.float64))
    return float(value)


def transform_1d(
    data,
    qc_data,
    qc_mask,
    input_coord,
    output_coord,
    transform,
    axis,
    input_missing_value=DEFAULT_MISSING,
    output_missing_value=DEFAULT_MISSING,
    # bin_average params
    input_bounds=None,
    output_bounds=None,
    weights=None,
    std_bad_max=np.inf,
    std_ind_max=np.inf,
    goodfrac_bad_min=0.0,
    goodfrac_ind_min=0.0,
    # interpolate / subsample params
    t_range=None,
):
    """Apply a 1D transform kernel along ``axis`` of an N-D array.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array of any shape.
    qc_data : numpy.ndarray or None
        Integer QC array with same shape as ``data``, or None (treated as
        all-zero / no QC).
    qc_mask : int, str, list[str], or None
        Bitmask, assessment name, or None. An assessment name is matched
        against ``qc_data`` metadata; None selects ``"Bad"`` when QC metadata
        is available and otherwise excludes no QC bits.
    input_coord : numpy.ndarray
        1-D coordinate values for the transform dimension (length == ``data.shape[axis]``).
    output_coord : numpy.ndarray
        1-D target coordinate values (length == output size along ``axis``).
    transform : {"bin_average", "interpolate", "subsample"}
        Which transform kernel to apply.
    axis : int
        Axis of ``data`` along which to apply the transform.
    input_missing_value : float
        Sentinel value for missing input data.
    output_missing_value : float
        Sentinel value to fill missing output data.
    input_bounds : numpy.ndarray, optional
        Shape ``(ni, 2)`` array of [start, end] bounds for each input bin.
        Used for ``"bin_average"``; inferred from midpoints if None.
    output_bounds : numpy.ndarray, optional
        Shape ``(nt, 2)`` array of [start, end] bounds for each output bin.
        Used for ``"bin_average"``; inferred from midpoints if None.
    weights : numpy.ndarray, optional
        Per-input-sample weights for ``"bin_average"``. Defaults to ones.
    std_bad_max : float
        Stdev threshold above which output is flagged ``QC_BAD_STD``.
    std_ind_max : float
        Stdev threshold above which output is flagged ``QC_INDETERMINATE_STD``.
    goodfrac_bad_min : float
        Coverage fraction below which output is flagged ``QC_BAD_GOODFRAC``.
    goodfrac_ind_min : float
        Coverage fraction below which output is flagged ``QC_INDETERMINATE_GOODFRAC``.
    t_range : float or numpy.timedelta64, optional
        Max distance for interpolate/subsample; a timedelta is converted to
        nanoseconds to match a datetime coordinate. Defaults to median input
        spacing.

    Returns
    -------
    output_data : numpy.ndarray
        Transformed data array; same shape as ``data`` except along ``axis``
        where the size equals ``len(output_coord)``.
    output_qc : numpy.ndarray
        Integer QC array with same shape as ``output_data``.

    """
    data = np.asarray(data, dtype=np.float64)
    ni = len(input_coord)
    nt = len(output_coord)

    # The kernels do arithmetic directly on the coordinate arrays, so reduce
    # datetime-like coordinates to numbers up front for every transform.
    # bin_average used to get this only incidentally, via _get_bounds().
    input_coord = _to_numeric(input_coord)
    output_coord = _to_numeric(output_coord)
    t_range = _to_numeric_scalar(t_range)

    if data.shape[axis] != ni:
        raise ValueError(
            f"data.shape[{axis}]={data.shape[axis]} does not match len(input_coord)={ni}"
        )

    if qc_data is None:
        qc_data = np.zeros(data.shape, dtype=np.int32)
    else:
        qc_data = np.asarray(qc_data, dtype=np.int32)

    # Move the transform axis to position 0 for easy iteration
    data_t = np.moveaxis(data, axis, 0)  # (ni, ...)
    qc_t = np.moveaxis(qc_data, axis, 0)  # (ni, ...)

    other_shape = data_t.shape[1:]
    n_other = int(np.prod(other_shape)) if other_shape else 1

    # Flatten to (ni, n_other) for uniform iteration
    data_2d = data_t.reshape(ni, n_other)
    qc_2d = qc_t.reshape(ni, n_other)

    out_2d = np.full((nt, n_other), output_missing_value, dtype=np.float64)
    qc_out_2d = np.zeros((nt, n_other), dtype=np.int32)

    if transform == 'bin_average':
        in_start, in_end = _get_bounds(input_coord, input_bounds)
        out_start, out_end = _get_bounds(output_coord, output_bounds)
        for col in range(n_other):
            stdev = np.full(nt, output_missing_value)
            coverage = np.full(nt, output_missing_value)
            _bin_average_1d(
                array=data_2d[:, col],
                qc_array=qc_2d[:, col],
                qc_mask=qc_mask,
                index_start=in_start,
                index_end=in_end,
                weights=weights,
                ni=ni,
                output=out_2d[:, col],
                qc_output=qc_out_2d[:, col],
                target_start=out_start,
                target_end=out_end,
                nt=nt,
                input_missing_value=input_missing_value,
                output_missing_value=output_missing_value,
                rmet=[stdev, coverage],
                std_bad_max=std_bad_max,
                std_ind_max=std_ind_max,
                goodfrac_bad_min=goodfrac_bad_min,
                goodfrac_ind_min=goodfrac_ind_min,
            )

    elif transform == 'interpolate':
        if t_range is None:
            t_range = _infer_t_range(input_coord)
        for col in range(n_other):
            dist_1 = np.full(nt, output_missing_value)
            dist_2 = np.full(nt, output_missing_value)
            _bilinear_interpolate_1d(
                array=data_2d[:, col],
                qc_array=qc_2d[:, col],
                qc_mask=qc_mask,
                index=input_coord,
                ni=ni,
                output=out_2d[:, col],
                qc_output=qc_out_2d[:, col],
                target=output_coord,
                nt=nt,
                input_missing_value=input_missing_value,
                output_missing_value=output_missing_value,
                rmet=[dist_1, dist_2],
                t_range=t_range,
            )

    elif transform == 'subsample':
        if t_range is None:
            t_range = _infer_t_range(input_coord)
        for col in range(n_other):
            distance = np.full(nt, output_missing_value)
            _subsample_1d(
                array=data_2d[:, col],
                qc_array=qc_2d[:, col],
                qc_mask=qc_mask,
                index=input_coord,
                ni=ni,
                output=out_2d[:, col],
                qc_output=qc_out_2d[:, col],
                target=output_coord,
                nt=nt,
                input_missing_value=input_missing_value,
                output_missing_value=output_missing_value,
                rmet=[distance],
                t_range=t_range,
            )

    else:
        raise ValueError(
            f"Unknown transform '{transform}'. "
            "Must be one of: 'bin_average', 'interpolate', 'subsample'."
        )

    # Reshape back to (nt, *other_shape) then move axis back
    out_nd = out_2d.reshape((nt,) + other_shape)
    qc_nd = qc_out_2d.reshape((nt,) + other_shape)

    output_data = np.moveaxis(out_nd, 0, axis)
    output_qc = np.moveaxis(qc_nd, 0, axis)

    return output_data, output_qc


def _get_bounds(coord, bounds):
    """Return (start, end) bound arrays for a coordinate.

    If ``bounds`` is provided (shape ``(n, 2)``), use it directly.
    Otherwise infer midpoint-based bounds from the coordinate values.

    Parameters
    ----------
    coord : numpy.ndarray
        1-D coordinate values.
    bounds : numpy.ndarray or None
        Shape ``(n, 2)`` array of [start, end] bounds, or None to infer them.

    Returns
    -------
    starts : numpy.ndarray
        Start coordinate of each bin.
    ends : numpy.ndarray
        End coordinate of each bin.

    """
    if bounds is not None:
        b = _to_numeric(bounds)
        return b[:, 0], b[:, 1]

    coord = _to_numeric(coord)
    n = len(coord)
    starts = np.empty(n)
    ends = np.empty(n)

    if n == 1:
        starts[0] = coord[0]
        ends[0] = coord[0]
        return starts, ends

    # Interior midpoints
    mids = 0.5 * (coord[:-1] + coord[1:])

    # Extrapolate edges
    starts[0] = coord[0] - (mids[0] - coord[0])
    starts[1:] = mids

    ends[:-1] = mids
    ends[-1] = coord[-1] + (coord[-1] - mids[-1])

    return starts, ends


def _get_missing_value(da):
    """Extract a missing value from DataArray encoding or attrs, with fallback."""
    for src in (da.encoding, da.attrs):
        for key in ('_FillValue', 'missing_value'):
            val = src.get(key)
            if val is not None:
                return float(val)
    return DEFAULT_MISSING


def apply_transform(
    data,
    target,
    dim,
    transform,
    qc=None,
    qc_mask=None,
    **kwargs,
):
    """Apply a 1D transform kernel to an :class:`xarray.DataArray` along ``dim``.

    Shared xarray-shape-handling implementation used by
    :func:`act.transform.bin_average.bin_average`,
    :func:`act.transform.interpolate.interpolate`, and
    :func:`act.transform.subsample.subsample`.

    Parameters
    ----------
    data : xarray.DataArray
        Input DataArray.
    target : xarray.DataArray or numpy.ndarray
        Target coordinate values.
    dim : str
        Name of the dimension along which to apply the transform.
    transform : {"bin_average", "interpolate", "subsample"}
        Which transform kernel to apply.
    qc : xarray.DataArray, optional
        Optional integer QC DataArray with same shape as ``data``.
    qc_mask : int, str, list[str], or None
        Integer bitmask, QC assessment name, or None. An assessment name is
        matched against ``qc.attrs['flag_assessments']``; None selects
        ``"Bad"`` when QC metadata is available and otherwise excludes no QC
        bits.
    **kwargs
        Additional keyword arguments forwarded to :func:`transform_1d`.

    Returns
    -------
    result : xarray.DataArray
        Transformed DataArray on the target coordinate.
    result_qc : xarray.DataArray
        Integer QC DataArray with same shape as ``result``, annotated with
        ARM-style ``flag_masks``/``flag_meanings``/``flag_assessments``/
        ``flag_comments`` and ``standard_name='quality_flag'`` (see
        :func:`act.transform.constants.add_arm_qc_attrs`).

    """
    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray with dims {data.dims}")

    input_coord = (
        data[dim].values if dim in data.coords else np.arange(data.sizes[dim], dtype=float)
    )

    if isinstance(target, xr.DataArray):
        output_coord = target.values
        target_da = target
    else:
        output_coord = np.asarray(target)
        # Keep datetime-like targets as-is so the output coordinate stays a
        # datetime; transform_1d reduces it to numbers for the kernels. Numeric
        # targets keep their historical float64 cast.
        if not _is_datetime_like_array(output_coord):
            output_coord = output_coord.astype(float)
        target_da = xr.DataArray(output_coord, dims=[dim], name=dim)

    if dim in data.coords:
        orig_coord = data[dim]
        target_da = target_da.copy()
        for k, v in orig_coord.attrs.items():
            if k not in target_da.attrs:
                target_da.attrs[k] = v
        for k, v in orig_coord.encoding.items():
            if k not in target_da.encoding:
                target_da.encoding[k] = v

    axis = data.dims.index(dim)
    missing = _get_missing_value(data)

    qc_arr = qc.values if qc is not None else None
    resolved_qc_mask = _resolve_qc_mask(qc, qc_mask)

    out_data, out_qc_arr = transform_1d(
        data=data.values,
        qc_data=qc_arr,
        qc_mask=resolved_qc_mask,
        input_coord=input_coord,
        output_coord=output_coord,
        transform=transform,
        axis=axis,
        input_missing_value=missing,
        output_missing_value=missing,
        **kwargs,
    )

    # Build output coords: replace the transformed dim, keep all others
    new_coords = {}
    for c in data.coords:
        if dim in data[c].dims:
            if c == dim:
                new_coords[dim] = target_da
        else:
            new_coords[c] = data[c]

    if dim not in new_coords:
        new_coords[dim] = target_da

    new_dims = data.dims

    result = xr.DataArray(
        out_data,
        dims=new_dims,
        coords=new_coords,
        attrs=data.attrs,
        name=data.name,
    )
    result.encoding.update(data.encoding)

    qc_name = f"qc_{data.name}" if data.name else 'qc'
    result_qc = xr.DataArray(
        out_qc_arr,
        dims=new_dims,
        coords=new_coords,
        name=qc_name,
    )
    add_arm_qc_attrs(result_qc)

    return result, result_qc


def _extract_bounds(coord, bounds_name, ds):
    """Return an (n, 2) bounds array for ``coord`` if a bounds variable exists, else None.

    Parameters
    ----------
    coord : xarray.DataArray
        Coordinate to look up bounds for.
    bounds_name : str or None
        Explicit name of the bounds variable, or None to look it up via
        ``coord.attrs['bounds']``.
    ds : xarray.Dataset or None
        Dataset that may contain the bounds variable.

    Returns
    -------
    numpy.ndarray or None
        Shape ``(n, 2)`` bounds array, or None if not found.

    """
    if bounds_name is None and hasattr(coord, 'attrs'):
        bounds_name = coord.attrs.get('bounds')
    if bounds_name is None or ds is None:
        return None
    if bounds_name in ds:
        return ds[bounds_name].values
    return None


def make_coord(start, stop, freq, name='time'):
    """Create a coordinate DataArray for datetimes or numeric steps.

    Parameters
    ----------
    start : int, float, str, numpy.datetime64, or pandas.Timestamp
        Start value (datetime-like or numeric).
    stop : int, float, str, numpy.datetime64, or pandas.Timestamp
        Stop value (datetime-like or numeric).
    freq : int, float, or str
        Frequency step. For datetimes, a pandas frequency string
        (e.g. ``'30min'``, ``'1h'``). For numeric coordinates, a float or
        int step size.
    name : str
        Name of the coordinate and dimension (default ``'time'``).

    Returns
    -------
    xarray.DataArray
        Coordinate array with ``name`` as its dimension and coordinate.

    Examples
    --------
    A datetime coordinate, from datetime-like endpoints and a pandas frequency
    string. The endpoints may be strings, ``numpy.datetime64``, or
    ``pandas.Timestamp``.

    .. code-block:: python

        target_time = act.transform.make_coord('2023-01-01', '2023-01-02', '30min')

    A numeric coordinate, from numeric endpoints and a numeric step. Note that
    the numeric form excludes ``stop``, following ``numpy.arange``, while the
    datetime form includes it, following ``pandas.date_range``.

    .. code-block:: python

        target_height = act.transform.make_coord(0.0, 5000.0, 250.0, name='height')

    """

    def _is_datetime_like(val):
        if isinstance(val, (np.datetime64, pd.Timestamp)):
            return True
        if hasattr(val, 'year') and hasattr(val, 'month') and hasattr(val, 'day'):
            return True
        if isinstance(val, str):
            try:
                pd.to_datetime(val)
                return True
            except (ValueError, TypeError):
                return False
        return False

    if _is_datetime_like(start) or _is_datetime_like(stop):
        values = pd.date_range(start=start, end=stop, freq=freq).values
    else:
        values = np.arange(start, stop, freq)

    return xr.DataArray(values, dims=[name], coords={name: values}, name=name)


def transform_dataset(
    ds,
    target=None,
    dim=None,
    transform=None,
    qc_prefix='qc_',
    qc_mask=None,
    target_ds=None,
    per_var_transform=None,
    per_var_kwargs=None,
    **kwargs,
):
    """Apply a transform to every variable in ``ds`` that contains ``dim``.

    For each variable, if a companion QC variable named ``{qc_prefix}<var>``
    exists in the dataset it is used automatically.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset.
    target : xarray.DataArray or numpy.ndarray, optional
        Target coordinate values. Mutually exclusive with ``target_ds``.
    dim : str
        Dimension name along which to transform.
    transform : {"bin_average", "interpolate", "subsample"}
        Default transform to apply to every matching variable.
    qc_prefix : str
        Prefix used to find companion QC variables (default ``"qc_"``).
    qc_mask : int, str, list[str], or None
        Integer bitmask, QC assessment name, or None. An assessment name is
        matched against each QC companion's ``flag_assessments`` metadata;
        None selects ``"Bad"`` when metadata is available and otherwise
        excludes no QC bits.
    target_ds : xarray.Dataset, optional
        Source dataset containing the target coordinate.
    per_var_transform : dict, optional
        Mapping of variable name to a transform override.
    per_var_kwargs : dict, optional
        Mapping of variable name to keyword argument overrides.
    **kwargs
        Additional keyword arguments forwarded to the underlying transform
        (e.g. ``t_range``, ``std_bad_max``).

    Returns
    -------
    xarray.Dataset
        New dataset with transformed variables on the target coordinate.
        Each transformed variable has a companion QC variable included.

    See Also
    --------
    act.transform.bin_average : Bin-average a single DataArray.
    act.transform.interpolate : Interpolate a single DataArray.
    act.transform.subsample : Subsample a single DataArray.

    Examples
    --------
    Bin-average every time-dependent variable onto a 30-minute time base. Each
    variable's ``qc_<var>`` companion is found and honored automatically, and for
    ``bin_average`` the coordinate's CF ``bounds`` variable is picked up as
    ``input_bounds``. The bounds variable itself is not transformed.

    .. code-block:: python

        import act

        ds = act.io.arm.read_arm_netcdf(filename, cleanup_qc=True)
        target = act.transform.make_coord('2023-03-01', '2023-03-02', '30min')

        new_ds = act.transform.transform_dataset(
            ds, target=target, dim='time', transform='bin_average', qc_mask=4
        )

    One transform is rarely right for every variable in a file, which is the
    usual reason to use this instead of looping yourself. ``per_var_transform``
    overrides the transform for named variables and ``per_var_kwargs`` overrides
    their keyword arguments -- here a categorical present-weather code is
    subsampled rather than averaged, since averaging two codes yields a value
    that corresponds to no weather condition.

    .. code-block:: python

        new_ds = act.transform.transform_dataset(
            ds,
            target=target,
            dim='time',
            transform='bin_average',  # default for everything else
            qc_mask=4,
            per_var_transform={'pwd_pw_code_inst': 'subsample'},
            per_var_kwargs={'wspd_arith_mean': {'std_ind_max': 0.8}},
        )

    Transform onto another Dataset's coordinate instead of an explicit target,
    which is convenient for putting two datastreams on a common time base.
    ``target`` and ``target_ds`` are mutually exclusive.

    .. code-block:: python

        new_ds = act.transform.transform_dataset(
            ds, target_ds=other_ds, dim='time', transform='interpolate'
        )

    """
    if dim is None:
        raise ValueError("The 'dim' parameter is required.")
    if transform is None:
        raise ValueError("The 'transform' parameter is required.")

    if target is None and target_ds is None:
        raise ValueError("Must provide either 'target' or 'target_ds'")
    if target is not None and target_ds is not None:
        raise ValueError("Cannot provide both 'target' and 'target_ds'")

    if target_ds is not None:
        if dim not in target_ds:
            raise ValueError(f"Dimension '{dim}' not found in target_ds")
        target_da = target_ds[dim]
    else:
        target_da = target

    _transform_fn = {
        'bin_average': bin_average,
        'interpolate': interpolate,
        'subsample': subsample,
    }

    result_vars = {}
    per_var_transform = per_var_transform or {}
    per_var_kwargs = per_var_kwargs or {}

    # CF bounds variables describe the coordinate cells rather than measured
    # data, so they are consumed as bounds below and never transformed. They
    # are also not necessarily numeric -- read_arm_netcdf leaves them as
    # cftime objects -- so feeding them to a kernel would fail outright.
    bounds_vars = {
        v.attrs['bounds'] for v in ds.variables.values() if 'bounds' in getattr(v, 'attrs', {})
    }

    for name, da in ds.data_vars.items():
        if name in bounds_vars:
            continue

        if dim not in da.dims:
            result_vars[name] = da
            continue

        # Skip variables that are themselves QC companions
        if isinstance(name, str) and name.startswith(qc_prefix):
            continue

        qc_name = f'{qc_prefix}{name}'
        qc_da = ds[qc_name] if qc_name in ds else None

        var_transform = per_var_transform.get(name, transform)
        if var_transform not in _transform_fn:
            raise ValueError(
                f"Unknown transform '{var_transform}' for variable '{name}'. "
                f'Must be one of {list(_transform_fn)}'
            )
        fn = _transform_fn[var_transform]

        # Combine global kwargs and per-variable kwargs
        var_kwargs = kwargs.copy()
        if name in per_var_kwargs:
            var_kwargs.update(per_var_kwargs[name])

        # Auto-detect bounds if bin_average is used and bounds aren't explicitly provided
        if var_transform == 'bin_average':
            if 'input_bounds' not in var_kwargs and dim in ds.coords:
                ib = _extract_bounds(ds[dim], None, ds)
                if ib is not None:
                    var_kwargs['input_bounds'] = ib

            if 'output_bounds' not in var_kwargs:
                if target_ds is not None and dim in target_ds.coords:
                    ob = _extract_bounds(target_ds[dim], None, target_ds)
                    if ob is not None:
                        var_kwargs['output_bounds'] = ob

        out, out_qc = fn(da, target_da, dim, qc=qc_da, qc_mask=qc_mask, **var_kwargs)
        result_vars[name] = out
        result_vars[qc_name] = out_qc

    return xr.Dataset(result_vars, attrs=ds.attrs)


@xr.register_dataset_accessor('transform')
class Transform:
    """Xarray Dataset accessor exposing the ``act.transform`` kernels.

    Every method here forwards straight through to the corresponding
    module-level function (:func:`act.transform.bin_average`,
    :func:`act.transform.interpolate`, :func:`act.transform.subsample`,
    :func:`act.transform.transform_dataset`) using ``self._ds[var]`` in
    place of an explicit DataArray argument. No transform logic is
    duplicated here.

    Examples
    --------
    The accessor takes variable *names* and looks them up in the Dataset, where
    the module-level functions take :class:`xarray.DataArray` objects. Note that
    the QC argument changes name accordingly: ``qc_var_name`` here versus ``qc``
    on the functions.

    .. code-block:: python

        import act

        ds = act.io.arm.read_arm_netcdf(filename, cleanup_qc=True)
        target = act.transform.make_coord('2023-03-01', '2023-03-02', '30min')

        result, result_qc = ds.transform.bin_average(
            'temp_mean', target, dim='time', qc_var_name='qc_temp_mean', qc_mask=4
        )

    These two calls are equivalent; the accessor forwards straight to the
    function, so neither form is more capable than the other.

    .. code-block:: python

        result, result_qc = ds.transform.interpolate('temp_mean', target, dim='time')

        result, result_qc = act.transform.interpolate(ds['temp_mean'], target, dim='time')

    Any extra keyword arguments are passed through to the underlying function,
    and ``transform_dataset`` takes every argument except ``ds`` itself.

    .. code-block:: python

        result, result_qc = ds.transform.bin_average(
            'wspd_arith_mean', target, dim='time', std_ind_max=0.8
        )

        new_ds = ds.transform.transform_dataset(
            target=target, dim='time', transform='bin_average', qc_mask=4
        )

    """

    def __init__(self, ds):
        self._ds = ds

    def bin_average(self, var_name, target, dim, qc_var_name=None, qc_mask=None, **kwargs):
        """Bin-average ``self._ds[var_name]`` onto ``target``. See :func:`act.transform.bin_average`."""
        qc = self._ds[qc_var_name] if qc_var_name is not None else None
        return bin_average(self._ds[var_name], target, dim, qc=qc, qc_mask=qc_mask, **kwargs)

    def interpolate(self, var_name, target, dim, qc_var_name=None, qc_mask=None, **kwargs):
        """Interpolate ``self._ds[var_name]`` onto ``target``. See :func:`act.transform.interpolate`."""
        qc = self._ds[qc_var_name] if qc_var_name is not None else None
        return interpolate(self._ds[var_name], target, dim, qc=qc, qc_mask=qc_mask, **kwargs)

    def subsample(self, var_name, target, dim, qc_var_name=None, qc_mask=None, **kwargs):
        """Subsample ``self._ds[var_name]`` onto ``target``. See :func:`act.transform.subsample`."""
        qc = self._ds[qc_var_name] if qc_var_name is not None else None
        return subsample(self._ds[var_name], target, dim, qc=qc, qc_mask=qc_mask, **kwargs)

    def transform_dataset(self, **kwargs):
        """Apply a transform to every matching variable in this Dataset. See :func:`act.transform.transform_dataset`."""
        return transform_dataset(self._ds, **kwargs)
