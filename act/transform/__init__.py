"""
QC-aware, bounds-aware transformations (bin-averaging, interpolation, and
subsampling) for moving data between coordinate grids, e.g. resampling onto
a different time base.

Every transform is available two ways: as a plain function
(``act.transform.bin_average(...)``) that accepts and returns
:class:`xarray.DataArray` objects, and as an :class:`xarray.Dataset`
accessor method (``ds.transform.bin_average(...)``) that operates on a
named variable of a Dataset. The accessor forwards to the plain functions;
no logic is duplicated between the two.

The core numeric kernels are JIT-compiled with numba for performance. If
numba is unavailable or fails to compile, each kernel transparently falls
back to a slower pure-Python implementation and emits a warning; results
are numerically equivalent either way.

"""

import lazy_loader as lazy

# Import eagerly: each of these functions shares its name with the module
# that defines it (e.g. `bin_average.bin_average`), so a plain lazy_loader
# submodule/attr split would let a later `import act.transform.bin_average`
# elsewhere in the codebase clobber the `act.transform.bin_average`
# function with the module object. Importing them up front avoids that.
from .bin_average import bin_average  # noqa
from .interpolate import interpolate  # noqa
from .subsample import subsample  # noqa

__getattr__, _lazy_dir, _lazy_all = lazy.attach(
    __name__,
    submodules=['constants', 'driver'],
    submod_attrs={
        'constants': [
            'QC_BAD',
            'QC_INDETERMINATE',
            'QC_INTERPOLATE',
            'QC_EXTRAPOLATE',
            'QC_NOT_USING_CLOSEST',
            'QC_SOME_BAD_INPUTS',
            'QC_ZERO_WEIGHT',
            'QC_OUTSIDE_RANGE',
            'QC_ALL_BAD_INPUTS',
            'QC_BAD_STD',
            'QC_INDETERMINATE_STD',
            'QC_BAD_GOODFRAC',
            'QC_INDETERMINATE_GOODFRAC',
        ],
        'driver': ['Transform', 'make_coord', 'transform_dataset'],
    },
)
__all__ = sorted(set(_lazy_all) | {'bin_average', 'interpolate', 'subsample'})


def __dir__():
    """Return the full public surface, including the eagerly-imported transforms.

    ``lazy.attach`` builds its ``__dir__`` from the names it was asked to load
    lazily, so the three functions imported eagerly above would be absent. That
    matters beyond interactive tab-completion: ``sphinx.ext.autosummary`` falls
    back to ``dir()`` when generating a module's API page, so ``bin_average``,
    ``interpolate``, and ``subsample`` were silently omitted from the generated
    documentation. Report ``__all__`` instead, which already accounts for them.
    """
    return list(__all__)
