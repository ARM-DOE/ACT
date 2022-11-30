"""
ACT: The Atmospheric Community Toolkit
======================================

"""

import lazy_loader as lazy

# No more pandas warnings
from pandas.plotting import register_matplotlib_converters

from . import tests
from ._version import get_versions

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

# Import early so these classes are available to the object
from act.qc.qcfilter import QCFilter
from act.qc.qctests import QCTests
from act.qc import clean

# Version for source builds
vdict = get_versions()
__version__ = vdict['version']
