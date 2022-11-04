"""
ACT: The Atmospheric Community Toolkit
======================================

"""

import lazy_loader as lazy

# No more pandas warnings
from pandas.plotting import register_matplotlib_converters

from . import tests
from ._version import get_versions
from .qc import QCFilter, QCTests, clean

register_matplotlib_converters()

# Import the lazy loaded modules
submodules = [
    'corrections',
    'discovery',
    'io',
    'qc',
    'utils',
    'retrievals',
    'plotting',
]
__getattr__, __dir__, _ = lazy.attach(__name__, submodules)

# Version for source builds
vdict = get_versions()
__version__ = vdict['version']
