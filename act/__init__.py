"""
ACT: The Atmospheric Community Toolkit
======================================

"""

from . import io
from . import plotting
from . import corrections
from . import utils
from . import tests
from . import discovery
from . import retrievals
from . import qc
from ._version import get_versions

# No more pandas warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Version for source builds
vdict = get_versions()
__version__ = vdict["version"]
