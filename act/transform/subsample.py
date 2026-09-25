"""
Nearest-neighbor subsampling transformation kernel and its xarray-facing wrapper.

Selects the nearest input sample to each target coordinate point within a
specified range, accounting for QC flags. The core loop is numba-jitted
for speed with a pure-Python fallback (see :mod:`act.transform._numba_support`)
in case numba is unavailable or fails to compile.

"""

import numpy as np

from act.transform._numba_support import JitFallbackKernel
from act.transform.constants import (
    QC_ALL_BAD_INPUTS,
    QC_BAD,
    QC_INDETERMINATE,
    QC_NOT_USING_CLOSEST,
    QC_OUTSIDE_RANGE,
)


def _subsample_kernel_impl(
    array,
    qc_array,
    qc_mask,
    index,
    ni,
    output,
    qc_output,
    target,
    nt,
    input_missing_value,
    output_missing_value,
    distance,
    t_range,
):
    status = -1

    j = 0
    smallest_d_last_good_value = 0.0
    while j < nt:
        qc_output[j] = 0
        dist = 1e25
        smallest_d = dist
        it = -1

        i = 0
        while (i < ni) and (index[i] < target[j] - t_range):
            i += 1

        if i == ni:
            status = 1
            while j < nt:
                qc_output[j] = 0
                qc_output[j] |= QC_OUTSIDE_RANGE
                qc_output[j] |= QC_BAD
                output[j] = output_missing_value
                distance[j] = output_missing_value
                j += 1
            break

        first_iteration_of_while_loop = 1
        while i < ni:
            d = abs(index[i] - target[j])
            if d > t_range:
                break
            if d < smallest_d:
                smallest_d = d

            if (j != 0) and first_iteration_of_while_loop == 1:
                if index[i] > target[j]:
                    smallest_d = smallest_d_last_good_value

            if (
                d < dist
                and array[i] != input_missing_value
                and not (qc_array[i] & qc_mask)
                and np.isfinite(array[i])
            ):
                dist = d
                it = i

            if d > dist and it > 0:
                break

            first_iteration_of_while_loop = 0
            i += 1

        if it < 0:
            output[j] = output_missing_value
            distance[j] = output_missing_value
            status = 1
            qc_output[j] |= QC_ALL_BAD_INPUTS
            qc_output[j] |= QC_BAD

            if i == ni:
                j += 1
                while j < nt:
                    output[j] = output_missing_value
                    distance[j] = output_missing_value
                    qc_output[j] = 0
                    qc_output[j] |= QC_BAD
                    if target[j] < index[ni - 1] + t_range:
                        qc_output[j] |= QC_ALL_BAD_INPUTS
                    else:
                        qc_output[j] |= QC_OUTSIDE_RANGE
                    j += 1
                break

            j += 1
            continue

        output[j] = array[it]
        smallest_d_last_good_value = smallest_d
        distance[j] = index[it] - target[j]

        if (qc_array[it] & ~qc_mask) != 0:
            qc_output[j] |= QC_INDETERMINATE

        if dist > smallest_d:
            qc_output[j] |= QC_NOT_USING_CLOSEST

        j += 1

    return status


_subsample_kernel = JitFallbackKernel(_subsample_kernel_impl, 'subsample', cache=True)


def _subsample_1d(
    array,
    qc_array,
    qc_mask,
    index,
    ni,
    output,
    qc_output,
    target,
    nt,
    input_missing_value,
    output_missing_value,
    rmet,
    t_range,
):
    """Subsample ``array`` onto ``target`` coordinate points by nearest-neighbor lookup.

    This is the raw-numpy kernel-level function used internally by
    :func:`act.transform.driver.transform_1d`. Most users should call
    :func:`act.transform.subsample.subsample` instead, which operates on
    :class:`xarray.DataArray` objects.

    Parameters
    ----------
    array : numpy.ndarray
        Input data values (length ``ni``).
    qc_array : numpy.ndarray
        Integer QC flags for each input sample (length ``ni``).
    qc_mask : int
        Bitmask; bits set here are treated as bad.
    index : numpy.ndarray
        Input coordinate values (length ``ni``).
    ni : int
        Number of input samples.
    output : numpy.ndarray
        Pre-allocated output array (length ``nt``); modified in place.
    qc_output : numpy.ndarray
        Pre-allocated integer QC output array (length ``nt``); modified in place.
    target : numpy.ndarray
        Target coordinate values (length ``nt``).
    nt : int
        Number of target points.
    input_missing_value : float
        Sentinel for missing input.
    output_missing_value : float
        Sentinel to fill missing output.
    rmet : list of numpy.ndarray
        ``[distance]`` - pre-allocated signed-distance metric array (length ``nt``).
    t_range : float
        Maximum coordinate distance to search for a nearest neighbor.

    Returns
    -------
    int
        0 or 1 on success (1 means some points were outside range), negative on error.

    """
    return _subsample_kernel(
        array,
        qc_array,
        qc_mask,
        index,
        ni,
        output,
        qc_output,
        target,
        nt,
        input_missing_value,
        output_missing_value,
        rmet[0],
        t_range,
    )


def subsample(data, target, dim, qc=None, qc_mask=None, t_range=None):
    """Nearest-neighbor subsample ``data`` onto ``target`` coordinate values along ``dim``.

    Parameters
    ----------
    data : xarray.DataArray
        Input DataArray.
    target : xarray.DataArray or numpy.ndarray
        Target coordinate values to subsample onto.
    dim : str
        Name of the dimension along which to apply the transform.
    qc : xarray.DataArray, optional
        Optional integer QC DataArray with same shape as ``data``.
    qc_mask : int, str, list[str], or None
        Integer bitmask, QC assessment name, or None. An assessment name is
        matched against ``qc.attrs['flag_assessments']``; None selects
        ``"Bad"`` when QC metadata is available and otherwise excludes no QC
        bits.
    t_range : float or numpy.timedelta64, optional
        Maximum distance from a target point to an input point for selection.
        May be given as a timedelta when ``dim`` is a datetime coordinate.
        Defaults to the median spacing of input coordinate values.

    Returns
    -------
    result : xarray.DataArray
        Subsampled DataArray on the target coordinate.
    result_qc : xarray.DataArray
        Integer QC DataArray with same shape as ``result``, with CF
        ``flag_masks``/``flag_meanings``/``flag_assessments`` attributes set.

    See Also
    --------
    act.transform.bin_average : Weighted average over output bins.
    act.transform.interpolate : Linear interpolation onto the target coordinate.
    act.transform.transform_dataset : Apply a transform to every variable in a Dataset.

    Examples
    --------
    Select the nearest sample to each target point. Because subsampling returns
    a value that was actually observed rather than a computed one, it is the
    right choice for codes, flags, states, and other categorical quantities that
    averaging or interpolating would render meaningless.

    .. code-block:: python

        import act

        ds = act.io.arm.read_arm_netcdf(filename, cleanup_qc=True)
        target = act.transform.make_coord('2023-03-01', '2023-03-02', '30min')

        result, result_qc = act.transform.subsample(
            ds['pwd_pw_code_inst'], target, dim='time'
        )

    Honor input QC and cap how far the search may reach. When the nearest sample
    is excluded by ``qc_mask`` a farther one is used instead, and the output is
    flagged ``QC_NOT_USING_CLOSEST`` so the substitution is visible. When the
    coordinate is a datetime, ``t_range`` may be a timedelta; it defaults to the
    median input spacing.

    .. code-block:: python

        import numpy as np
        from act.transform.constants import QC_NOT_USING_CLOSEST

        result, result_qc = act.transform.subsample(
            ds['pwd_pw_code_inst'],
            target,
            dim='time',
            qc=ds['qc_pwd_pw_code_inst'],
            qc_mask=4,
            t_range=np.timedelta64(5, 'm'),
        )
        substituted = (result_qc.values & QC_NOT_USING_CLOSEST).astype(bool)

    """
    from act.transform.driver import apply_transform

    return apply_transform(
        data,
        target,
        dim,
        'subsample',
        qc=qc,
        qc_mask=qc_mask,
        t_range=t_range,
    )
