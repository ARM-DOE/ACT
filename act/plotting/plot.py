# Import third party libraries
import matplotlib.pyplot as plt
import datetime as dt
import astral
import numpy as np

# Import Local Libs
from . import common
from ..utils import datetime_utils as dt_utils
from ..utils import data_utils


class display(object):
    """
    A class for handing the display of ARM Datasets. The class stores
    the dataset to be plotted

    Attributes
    ----------
    fields: dict
        The dictionary containing the fields inside the ARM dataset. Each field
        has a key that links to an xarray DataArray object.
    ds: str
        The name of the datastream.
    file_dates: list
        The dates of each file being display
    plots: list
        The list of plots handled (currently not supported).
    plot_vars: list
        The list of variables being plotted.
    cbs: list
        The list of colorbar handles.

    """

    def __init__(self, arm_obj):
        """Initialize Object"""
        self._arm = arm_obj
        self.fields = arm_obj.variables
        self.ds = str(arm_obj['ds'].values)
        self.file_dates = arm_obj.file_dates.values

        self.plots = []
        self.plot_vars = []
        self.cbs = []

    def day_night_background(self, ax=None, fig=None):
        """
        Colorcodes the background according to sunrise/sunset

        Parameters
        ----------
        ax: matplotlib axis handle
            Axis handle to plot the bacground on. Set to None to use the
            current axis.
        fig: matplotlib figure handle
            Figure to plot the background on. Set to None to use the current
            figure handle.

        """
        # Get File Dates
        file_dates = self._arm.file_dates.data

        all_dates = dt_utils.dates_between(file_dates[-1], file_dates[0])

        # Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)

        # initialize the plot to a gray background for total darkness
        rect = ax.patch
        rect.set_facecolor('0.85')

        # Initiate Astral Instance
        a = astral.Astral()
        if self._arm.lat.data.size > 1:
            lat = self._arm.lat.data[0]
            lon = self._arm.lon.data[0]
        else:
            lat = float(self._arm.lat.data)
            lon = float(self._arm.lon.data)

        for f in all_dates:
            sun = a.sun_utc(f, lat, lon)

            # add yellow background for specified time period
            ax.axvspan(sun['sunrise'], sun['sunset'], facecolor='#FFFFCC')

            # add local solar noon line
            ax.axvline(x=sun['noon'], linestyle='--', color='y')

    def set_xrng(self, xrng, ax=None, fig=None):
        """
        Sets the x range of the plot.

        Parameters
        ----------
        xrng: 2 number array
            The x limits of the plot.
        ax: matplotlib axis handle
            Axis handle to plot the bacground on. Set to None to use the
            current axis.
        fig: matplotlib figure handle
            Figure to plot the background on. Set to None to use the current
            figure handle.

        """
        # Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)
        ax.set_xlim(xrng)
        self.xrng = xrng

    def set_yrng(self, yrng, ax=None, fig=None):
        """
        Sets the y range of the plot.

        Parameters
        ----------
        yrng: 2 number array
            The y limits of the plot.
        ax: matplotlib axis handle
            Axis handle to plot the bacground on. Set to None to use the
            current axis.
        fig: matplotlib figure handle
            Figure to plot the background on. Set to None to use the current
            figure handle.

        """
        # Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)
        ax.set_ylim(yrng)
        self.yrng = yrng

    def add_colorbar(self, mappable, title=None, ax=None, fig=None):
        """
        Adds a colorbar to the plot

        Parameters
        ----------
        mappable: matplotlib mappable
            The mappable to base the colorbar on.
        title: str
            The title of the colorbar. Set to None to have no title.
        ax: matplotlib axis handle
            Axis handle to plot the bacground on. Set to None to use the
            current axis.
        fig: matplotlib figure handle
            Figure to plot the background on. Set to None to use the current
            figure handle.

        Returns
        -------
        cbar: matplotlib colorbar handle
            The handle to the matplotlib colorbar.
        """
        # Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)
        # Give the colorbar it's own axis so the 2D plots line up with 1D
        box = ax.get_position()
        pad, width = 0.01, 0.01
        cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
        cbar = plt.colorbar(mappable, cax=cax)
        cbar.ax.set_ylabel(title, rotation=270, fontsize=8, labelpad=3)
        cbar.ax.tick_params(labelsize=6)

        return cbar

    def plot(self, field, ax=None, fig=None,
             cmap=None, cbmin=None, cbmax=None, set_title=None,
             add_nan=False, **kwargs):
        """
        Makes the plot

        Parameters
        ----------
        mappable: matplotlib mappable
            The mappable to base the colorbar on.
        title: str
            The title of the colorbar. Set to None to have no title.
        ax: matplotlib axis handle
            Axis handle to plot the bacground on. Set to None to use the
            current axis.
        fig: matplotlib figure handle
            Figure to plot the background on. Set to None to use the current
            figure handle.
        cmap: matplotlib colormap
            The colormap to use/
        cbmin: float
            The minimum for the colorbar.
        cbmax: float
            The maximum for the colorbar.
        set_title: str
            The title for the plot.
        add_nan: bool
            Set to True to fill in data gaps with NaNs.
        kwargs: dict
            The keyword arguments for plt.plot
        """

        # Get data and dimensions
        data = self._arm[field]
        dim = list(self._arm[field].dims)
        xdata = self._arm[dim[0]]
        ytitle = ''.join(['(', data.attrs['units'], ')'])
        if len(dim) > 1:
            ydata = self._arm[dim[1]]
            units = ytitle
            ytitle = ''.join(['(', ydata.attrs['units'], ')'])
        else:
            ydata = None

        # Get the current plotting axis, add day/night background and plot data
        ax, fig = common.parse_ax_fig(ax, fig)

        if ydata is None:
            self.day_night_background()
            ax.plot(xdata, data, '.')
        else:
            # Add in nans to ensure the data are not streaking
            if add_nan is True:
                xdata, data = data_utils.add_in_nan(xdata, data)
            mesh = ax.pcolormesh(xdata, ydata, data.transpose(),
                                 cmap=cmap, vmax=cbmax,
                                 vmin=cbmin, edgecolors='face')

        # Set Title
        if set_title is None:
            set_title = ' '.join([self.ds, field, 'on', self.file_dates[0]])

        plt.title(set_title)

        # Set YTitle
        ax.set_ylabel(ytitle)

        # Set X Limit
        if hasattr(self, 'xrng'):
            self.set_xrng(self.xrng)
        else:
            self.xrng = [xdata.min().values, xdata.max().values]
            self.set_xrng(self.xrng)

        # Set Y Limit
        if hasattr(self, 'yrng'):
            self.set_yrng(self.yrng)

        # Set X Format
        days = (self.xrng[1] - self.xrng[0]) / np.timedelta64(1, 'D')
        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)

        if ydata is not None:
            self.add_colorbar(mesh, title=units)
