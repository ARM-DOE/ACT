"""
act.plotting.act_cmap
=========================
Colorblind friendly colormaps
.. autosummary::
    :toctree: generated/
    _generate_cmap
Available colormaps, these
colormaps are available within matplotlib with names act_COLORMAP':
    * HomeyerRainbow
"""

import matplotlib as mpl
import matplotlib.colors as colors

from _act_cmap import datad, yuv_rainbow_24


def _generate_cmap(name, lutsize):
    """Generates the requested cmap from its name *name*.  The lut size is
    *lutsize*."""

    spec = datad[name]

    # Generate the colormap object.
    if 'red' in spec:
        return colors.LinearSegmentedColormap(name, spec, lutsize)
    else:
        return colors.LinearSegmentedColormap.from_list(name, spec, lutsize)

cmap_d = dict()

# register the colormaps so that they can be accessed with the names act_XXX
for name, cmap in cmap_d.items():
    full_name = 'act_' + name
    mpl.cm.register_cmap(name=full_name, cmap=cmap)