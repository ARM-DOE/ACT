"""
Available colormaps (reversed versions also provided), these
colormaps are available within matplotlib with names act_COLORMAP':

    * HomeyerRainbow

"""

from __future__ import print_function, division

import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors as colors

from ._act_cmap import datad, yuv_rainbow_24


def _reverser(f):
    """ perform reversal. """
    def freversed(x):
        """ f specific reverser. """
        return f(1 - x)
    return freversed


def revcmap(data):
    """ Can only handle specification *data* in dictionary format. """
    data_r = {}
    for key, val in data.items():
        if callable(val):
            valnew = _reverser(val)
            # This doesn't work: lambda x: val(1-x)
            # The same "val" (the first one) is used
            # each time, so the colors are identical
            # and the result is shades of gray.
        else:
            # Flip x and exchange the y values facing x = 0 and x = 1.
            valnew = [(1.0 - x, y1, y0) for x, y0, y1 in reversed(val)]
        data_r[key] = valnew
    return data_r


def _reverse_cmap_spec(spec):
    """ Reverses cmap specification *spec*, can handle both dict and tuple
    type specs. """

    if isinstance(spec, dict) and 'red' in spec.keys():
        return revcmap(spec)
    else:
        revspec = list(reversed(spec))
        if len(revspec[0]) == 2:    # e.g., (1, (1.0, 0.0, 1.0))
            revspec = [(1.0 - a, b) for a, b in revspec]
        return revspec


def _generate_cmap(name, lutsize):
    """ Generates the requested cmap from it's name *name*. The lut size is
    *lutsize*. """

    spec = datad[name]

    # Generate the colormap object.
    if isinstance(spec, dict) and 'red' in spec.keys():
        return colors.LinearSegmentedColormap(name, spec, lutsize)
    else:
        return colors.LinearSegmentedColormap.from_list(name, spec, lutsize)


cmap_d = dict()

LUTSIZE = mpl.rcParams['image.lut']

# need this list because datad is changed in loop
_cmapnames = list(datad.keys())

# Generate the reversed specifications ...

for cmapname in _cmapnames:
    spec = datad[cmapname]
    spec_reversed = _reverse_cmap_spec(spec)
    datad[cmapname + '_r'] = spec_reversed

# Precache the cmaps with ``lutsize = LUTSIZE`` ...

# Use datad.keys() to also add the reversed ones added in the section above:
for cmapname in datad.keys():
    cmap_d[cmapname] = _generate_cmap(cmapname, LUTSIZE)

locals().update(cmap_d)

# register the colormaps so that they can be accessed with the names act_XXX
for name, cmap in cmap_d.items():
    full_name = 'act_' + name
    mpl.cm.register_cmap(name=full_name, cmap=cmap)
