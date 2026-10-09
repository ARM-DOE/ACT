"""Verify each transform kernel's pure-Python fallback matches its numba-jitted path.

``act.transform`` depends on numba for performance but must never fail to
import or crash if numba is missing or fails to compile. These tests force
each kernel onto its fallback path via ``JitFallbackKernel.set_force_fallback``
and check that results are numerically identical to the JIT path, and that
a warning is emitted.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import xarray as xr

import act
from act.transform.bin_average import _bin_average_kernel
from act.transform.interpolate import _bilinear_interpolate_kernel
from act.transform.subsample import _subsample_kernel


@pytest.fixture
def time_series():
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.array([0.0, 2.0, 4.0, np.nan, 8.0, 10.0])
    return xr.DataArray(values, coords={'time': time}, dims=['time'], name='temp')


@pytest.mark.parametrize(
    'kernel, transform_fn, kwargs',
    [
        (_bin_average_kernel, act.transform.bin_average, {}),
        (_bilinear_interpolate_kernel, act.transform.interpolate, {}),
        (_subsample_kernel, act.transform.subsample, {'t_range': 1.5}),
    ],
)
def test_fallback_matches_jit(time_series, kernel, transform_fn, kwargs):
    target = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    kernel.set_force_fallback(False)
    kernel._warned = False
    jit_result, jit_qc = transform_fn(time_series, target, dim='time', **kwargs)

    kernel.set_force_fallback(True)
    kernel._warned = False
    try:
        with pytest.warns(RuntimeWarning, match='falling back'):
            fallback_result, fallback_qc = transform_fn(time_series, target, dim='time', **kwargs)
    finally:
        kernel.set_force_fallback(False)
        kernel._warned = False

    np.testing.assert_array_equal(jit_result.values, fallback_result.values)
    np.testing.assert_array_equal(jit_qc.values, fallback_qc.values)


def test_fallback_warns_only_once(time_series):
    import warnings

    kernel = _bin_average_kernel
    target = np.array([0.5, 1.5, 2.5])

    kernel.set_force_fallback(True)
    kernel._warned = False
    try:
        with pytest.warns(RuntimeWarning, match='falling back'):
            act.transform.bin_average(time_series, target, dim='time')

        # Second call on the fallback path should not emit a fresh warning.
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            act.transform.bin_average(time_series, target, dim='time')
        assert len(records) == 0
    finally:
        kernel.set_force_fallback(False)
        kernel._warned = False


def test_import_falls_back_when_numba_missing():
    """act.transform must still import and produce correct results if numba itself is unimportable.

    Run in a subprocess so that blocking numba's import can't leak into
    other tests (numba, once imported, stays cached in sys.modules).
    """
    script = textwrap.dedent("""
        import builtins
        import sys
        import warnings

        _real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == 'numba' or name.startswith('numba.'):
                raise ImportError('numba blocked for this test')
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = _blocked_import
        for mod_name in list(sys.modules):
            if mod_name == 'numba' or mod_name.startswith('numba.') or mod_name.startswith('act'):
                del sys.modules[mod_name]

        import numpy as np
        import xarray as xr
        import act  # must not raise, even though numba is unimportable

        time = np.array([0.0, 1.0, 2.0, 3.0])
        da = xr.DataArray([0.0, 2.0, 4.0, 6.0], coords={'time': time}, dims=['time'], name='temp')
        target = np.array([0.5, 1.5, 2.5])

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            result, qc = act.transform.interpolate(da, target, dim='time')

        assert result.shape == (3,)
        assert any(issubclass(r.category, RuntimeWarning) for r in records), records
        print('SUBPROCESS_OK')
        """)
    proc = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd='.',
    )
    assert proc.returncode == 0, f'stdout={proc.stdout}\nstderr={proc.stderr}'
    assert 'SUBPROCESS_OK' in proc.stdout
