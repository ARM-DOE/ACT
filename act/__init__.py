from . import io
from . import plotting
from . import corrections
from . import utils
from ._version import get_versions

# Version for source builds
vdict = get_versions()
__version__ = vdict["version"]

# Version for releases
# __version__ = "0.1"
