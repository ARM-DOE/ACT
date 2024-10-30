"""
ACT: The Atmospheric Community Toolkit
======================================

"""

import importlib.metadata as _importlib_metadata

import lazy_loader as lazy

# No more pandas warnings
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Import early so these classes are available to the object
from .qc import QCFilter, QCTests, clean  # noqa

# Import the lazy loaded modules
submodules = [
    'corrections',
    'discovery',
    'io',
    'qc',
    'utils',
    'retrievals',
    'plotting',
    'tests',
]
__getattr__, __dir__, _ = lazy.attach(__name__, submodules)

# Get the version
try:
    __version__ = _importlib_metadata.version("act-atmos")
except _importlib_metadata.PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"
