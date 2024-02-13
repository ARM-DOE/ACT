"""
ACT: The Atmospheric Community Toolkit
======================================

"""

import lazy_loader as lazy

# No more pandas warnings
from pandas.plotting import register_matplotlib_converters

from ._version import get_versions

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

# Version for source builds
vdict = get_versions()
__version__ = vdict['version']
