"""
act.plotting.WindRoseDisplay
----------------------------

Stores the class for WindRoseDisplay.

"""

import matplotlib.pyplot as plt
import numpy as np
import warnings

from .plot import Display
# Import Local Libs
from ..utils import datetime_utils as dt_utils


class WindRoseDisplay(Display):
    """
    A class for handing wind rose plots.

    This is inherited from the :func:`act.plotting.Display`
    class and has therefore has the same attributes as that class.
    See :func:`act.plotting.Display`
    for more information. There are no additional attributes or parameters
    to this class.

    Examples
    --------
    To create a WindRoseDisplay object, simply do:

    .. code-block :: python

        sonde_ds = act.io.armfiles.read_netcdf('sonde_data.nc')
        WindDisplay = act.plotting.WindRoseDisplay(sonde_ds, figsize=(8,10))

    """
    def __init__(self, obj, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(obj, subplot_shape, ds_name,
                         subplot_kw=dict(projection='polar'), **kwargs)

    def set_thetarng(self, trng=(0., 360.), subplot_index=(0,)):
        """
        Sets the theta range of the wind rose plot.

        Parameters
        ----------
        trng : 2-tuple
            The range (in degrees).
        subplot_index : 2-tuple
            The index of the subplot to set the degree range of.

        """
        if self.axes is not None:
            self.axes[subplot_index].set_thetamin(np.deg2rad(trng[0]))
            self.axes[subplot_index].set_thetamax(np.deg2rad(trng[1]))
            self.trng = trng
        else:
            raise RuntimeError(("Axes must be initialized before" +
                                " changing limits!"))

    def set_rrng(self, rrng, subplot_index=(0,)):
        """
        Sets the range of the radius of the wind rose plot.

        Parameters
        ----------
        rrng : 2-tuple
            The range for the plot radius (in %).
        subplot_index : 2-tuple
            The index of the subplot to set the radius range of.

        """
        if self.axes is not None:
            self.axes[subplot_index].set_thetamin(rrng[0])
            self.axes[subplot_index].set_thetamax(rrng[1])
            self.rrng = rrng
        else:
            raise RuntimeError(("Axes must be initialized before" +
                                " changing limits!"))

    def plot(self, dir_field, spd_field, dsname=None, subplot_index=(0,),
             cmap=None, set_title=None, num_dirs=20, spd_bins=None,
             tick_interval=3, **kwargs):
        """
        Makes the wind rose plot from the given dataset.

        Parameters
        ----------
        dir_field : str
            The name of the field representing the wind direction (in degrees).
        spd_field : str
            The name of the field representing the wind speed.
        dsname : str
            The name of the datastream to plot from. Set to None to
            let ACT automatically try to determine this.
        subplot_index : 2-tuple
            The index of the subplot to place the plot on.
        cmap : str or matplotlib colormap
            The name of the matplotlib colormap to use.
        set_title : str
            The title of the plot.
        num_dirs : int
            The number of directions to split the wind rose into.
        spd_bins : 1D array-like
            The bin boundaries to sort the wind speeds into.
        tick_interval : int
            The interval (in %) for the ticks on the radial axis.
        **kwargs : keyword arguments
            Additional keyword arguments will be passed into :func:plt.bar

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.

        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Get data and dimensions
        dir_data = self._arm[dsname][dir_field].values
        spd_data = self._arm[dsname][spd_field].values

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes(projection='polar')])
            self.fig.add_axes(self.axes[0])

        if spd_bins is None:
            spd_bins = np.linspace(0, np.nanmax(spd_data), 10)

        # Make the bins so that 0 degrees N is in the center of the first bin
        # We need to wrap around

        deg_width = 360. / num_dirs
        dir_bins_mid = np.linspace(0., 360. - 3 * deg_width / 2., num_dirs)
        wind_hist = np.zeros((num_dirs, len(spd_bins) - 1))

        for i in range(num_dirs):
            if i == 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "invalid value encountered in.*")
                    the_range = np.logical_or(dir_data < deg_width / 2.,
                                              dir_data > 360. - deg_width / 2.)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "invalid value encountered in.*")
                    the_range = np.logical_and(
                        dir_data >= dir_bins_mid[i] - deg_width / 2,
                        dir_data <= dir_bins_mid[i] + deg_width / 2)
            hist, bins = np.histogram(spd_data[the_range], spd_bins)
            wind_hist[i] = hist

        wind_hist = wind_hist / np.sum(wind_hist) * 100
        mins = np.deg2rad(dir_bins_mid)
        # Do the first level
        if 'units' in self._arm[dsname][spd_field].attrs.keys():
            units = self._arm[dsname][spd_field].attrs['units']
        else:
            units = ''
        the_label = ("%3.1f" % spd_bins[0] +
                     '-' + "%3.1f" % spd_bins[1] + " " + units)
        our_cmap = plt.cm.get_cmap(cmap)
        our_colors = our_cmap(np.linspace(0, 1, len(spd_bins)))

        bars = [self.axes[subplot_index].bar(mins, wind_hist[:, 0],
                                             label=the_label,
                                             width=0.8 * np.deg2rad(deg_width),
                                             color=our_colors[0],
                                             **kwargs)]
        for i in range(1, len(spd_bins) - 1):
            the_label = ("%3.1f" % spd_bins[i] +
                         '-' + "%3.1f" % spd_bins[i + 1] + " " + units)
            bars.append(self.axes[subplot_index].bar(
                mins, wind_hist[:, i], label=the_label,
                bottom=wind_hist[:, i - 1], width=0.8 * np.deg2rad(deg_width),
                color=our_colors[i], **kwargs))
        self.axes[subplot_index].legend()
        self.axes[subplot_index].set_theta_zero_location("N")
        self.axes[subplot_index].set_theta_direction(-1)
        # Set the ticks to be nice numbers
        tick_max = tick_interval * round(
            np.nanmax(np.cumsum(wind_hist, axis=1)) / tick_interval)
        rticks = np.arange(0, tick_max, tick_interval)
        rticklabels = [("%d" % x + '%') for x in rticks]
        self.axes[subplot_index].set_rticks(rticks)
        self.axes[subplot_index].set_yticklabels(rticklabels)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, 'on',
                                  dt_utils.numpy_to_arm_date(
                                      self._arm[dsname].time.values[0])])
        self.axes[subplot_index].set_title(set_title)
        return self.axes[subplot_index]
