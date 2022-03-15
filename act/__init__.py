"""
ACT: The Atmospheric Community Toolkit
======================================

"""

# No more pandas warnings
from pandas.plotting import register_matplotlib_converters

from . import corrections, discovery, io, plotting, qc, retrievals, tests, utils
from ._version import get_versions

register_matplotlib_converters()

# Version for source builds
vdict = get_versions()
__version__ = vdict['version']
