"""Internal helper for defensively wrapping the numba-jitted transform kernels.

``act.transform`` depends on numba for performance, but numba is treated as
not fully stable, so every JIT-compiled kernel used by this subpackage must
be able to fall back to a pure-Python/numpy implementation of the same logic
rather than crashing or preventing ``act.transform`` from being imported.
This module centralizes that "try JIT, else fall back" behavior so
``bin_average.py``, ``interpolate.py``, and ``subsample.py`` don't each
reimplement it.

"""

import warnings

try:
    from numba import njit as _njit

    _NUMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    _njit = None
    _NUMBA_IMPORT_ERROR = exc


class JitFallbackKernel:
    """Callable that prefers a numba-jitted kernel, falling back to plain Python.

    On each call, the numba-compiled version of ``python_fn`` is tried
    first. If numba could not be imported, could not compile ``python_fn``,
    or raises while running, the call transparently falls back to
    ``python_fn`` itself and a warning is emitted (once per instance) so
    a user knows they are on the slow path.

    Parameters
    ----------
    python_fn : callable
        Pure Python/numpy implementation of the kernel. Always correct;
        used directly when numba is unavailable or compilation/execution
        fails, and used as the source for JIT compilation otherwise.
    name : str
        Kernel name, used in the fallback warning message.
    **njit_kwargs
        Keyword arguments forwarded to ``numba.njit`` (e.g. ``cache=True``).

    """

    def __init__(self, python_fn, name, **njit_kwargs):
        self._python_fn = python_fn
        self._name = name
        self._force_fallback = False
        self._warned = False
        self._jit_fn = None
        self._fallback_reason = _NUMBA_IMPORT_ERROR
        if _njit is not None:
            try:
                self._jit_fn = _njit(**njit_kwargs)(python_fn)
            except Exception as exc:  # pragma: no cover - depends on numba internals
                self._fallback_reason = exc

    def set_force_fallback(self, force=True):
        """Force this kernel onto the pure-Python path.

        Intended for tests that need to exercise the fallback path
        deterministically without uninstalling numba.
        """
        self._force_fallback = force

    def __call__(self, *args, **kwargs):
        if self._jit_fn is not None and not self._force_fallback:
            try:
                return self._jit_fn(*args, **kwargs)
            except Exception as exc:
                # Compilation/execution failed for these arguments; stop
                # retrying the JIT path for this kernel and fall back.
                self._fallback_reason = exc
                self._jit_fn = None
        reason = 'forced fallback for testing' if self._force_fallback else self._fallback_reason
        self._warn(reason)
        return self._python_fn(*args, **kwargs)

    def _warn(self, reason):
        if self._warned:
            return
        self._warned = True
        warnings.warn(
            f"act.transform: numba JIT unavailable or failed for the "
            f"'{self._name}' kernel ({reason}); falling back to a slower "
            "pure-Python implementation. Results are unaffected.",
            RuntimeWarning,
            stacklevel=3,
        )
