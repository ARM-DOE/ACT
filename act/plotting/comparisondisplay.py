""" Module for Plotting the Comparison of Datastreams and Variables. """

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..utils import datetime_utils as dt_utils
from .plot import Display


class ComparisonDisplay(Display):
    """
    This class is used to make plots for comparing datastreams
    and variables. It is inherited from Display and therefore 
    contains all of Display's attributes and methods.

    Examples
    --------
    To create a ComparisonDisplay with 3 rows, simply do:

    .. code-block:: python

        ds = act.read_netcdf(the_file)
        disp = act.plotting.ComparisonDisplay(ds, subplot_shape=(3,), figsize=(15, 5))

    The ComparisonDisplay constructor takes in the same keyword arguments as
    plt.subplots. For more information on the plt.subplots keyword arguments,
    see the `matplotlib documentation
    <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html>`_.
    If no subplot_shape is provided, then no figure or axis will be created
    until add_subplots or plots is called.

    """

    def __init__(self, ds, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(ds, subplot_shape, ds_name, **kwargs)

    def set_xrng(self, xrng, subplot_index=(0,)):
        """
        Sets the x range of the plot.

        Parameters
        ----------
        xrng : 2 number array
            The x limits of the plot.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.

        """
        if self.axes is None:
            raise RuntimeError('set_xrng requires the plot to be displayed.')

        if not hasattr(self, 'xrng') and len(self.axes.shape) == 2:
            self.xrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2), dtype='datetime64[D]')
        elif not hasattr(self, 'xrng') and len(self.axes.shape) == 1:
            self.xrng = np.zeros((self.axes.shape[0], 2), dtype='datetime64[D]')

        self.axes[subplot_index].set_xlim(xrng)
        self.xrng[subplot_index, :] = np.array(xrng)

    def set_yrng(self, yrng, subplot_index=(0,)):
        """
        Sets the y range of the plot.

        Parameters
        ----------
        yrng : 2 number array
            The y limits of the plot.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.

        """
        if self.axes is None:
            raise RuntimeError('set_yrng requires the plot to be displayed.')

        if not hasattr(self, 'yrng') and len(self.axes.shape) == 2:
            self.yrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2))
        elif not hasattr(self, 'yrng') and len(self.axes.shape) == 1:
            self.yrng = np.zeros((self.axes.shape[0], 2))

        if yrng[0] == yrng[1]:
            yrng[1] = yrng[1] + 1

        self.axes[subplot_index].set_ylim(yrng)
        self.yrng[subplot_index, :] = yrng

    def set_ratio_line(self, subplot_index=(0,)):
        """
        Sets the 1:1 ratio line.

        Parameters
        ----------
        xfield : str
            Name of the field displayed on the x-axis
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.

        """
        if self.axes is None:
            raise RuntimeError('set_ratio_line requires the plot to be displayed.')
        # Define the xticks of the figure
        xlims = self.axes[subplot_index].get_xticks()
        ratio = np.linspace(xlims[0], xlims[-1])
        self.axes[subplot_index].plot(ratio, ratio, 'k--')

    def _get_data(self, dsname, fields):
        if isinstance(fields, str):
            fields = [fields]
        return self._ds[dsname][fields].dropna('time')

    def scatter(
        self,
        x_field,
        y_field,
        m_field=None,
        marker=None,
        cmap=None,
        dsname=None,
        subplot_index=(0,),
        set_title=None,
        cbar_label=None,
        best_fit=False,
        correlation=False,
        **kwargs,
    ):
        """
        This procedure will produce a scatter plot from 2 variables.

        Parameters
        ----------
        x_field : str
            The name of the field to display on the X axis.
        y_field : str
            The name of the field to display on the Y axis.
        m_field : str
            The name of the field to display on the markers.
        marker : str
            The style of marker to use following matplotlib.markers
            Defaults are points
        cmap : str
            Color map to use for colorbar.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        subplot_index : tuple
            The subplot index to place the plot in
        set_title : str
            The title of the plot.
        cbar_label : str
            The title of the colorbar (if needed)
        best_fit : Boolean
            Boolean flag to calculate and display best fit linear line
        correlation : Boolean
            Boolean flag to calculate persion correlation coefficient and display

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.scatter`.

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        if m_field is None:
            mdata = None
            ds = self._get_data(dsname, [x_field, y_field])
            xdata, ydata = ds[x_field], ds[y_field]
        else:
            ds = self._get_data(dsname, [x_field, y_field, m_field])
            xdata, ydata, mdata = ds[x_field], ds[y_field], ds[m_field]

        # Define the x-axis label. If units are avaiable, plot. 
        if 'units' in xdata.attrs:
            xtitle = x_field + ''.join([' (', xdata.attrs['units'], ')'])
        else:
            xtitle = x_field
        
        # Define the y-axis label. If units are available, plot
        if 'units' in ydata.attrs:
            ytitle = y_field + ''.join([' (', ydata.attrs['units'], ')'])
        else:
            ytitle = y_field

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        # Define the axes for the figure
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        # Define the default marker style of the plot
        if marker is None:
            marker = 'o'

        # Define the colorbar map if a marker field is added
        if mdata is not None: 
            if cmap is None:
                cmap = 'act_HomeyerRainbow'
            else:
                cmap = plt.get_cmap(cmap)
        
        # Display the scatter plot, pass keyword args for unspecified attributes
        sc = self.axes[subplot_index].scatter(xdata, 
                                              ydata, 
                                              c=mdata, 
                                              marker=marker,
                                              cmap=cmap,
                                              **kwargs)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )

        # Check to see if a colorbar label was set
        if mdata is not None:
            if cbar_label is None:
                # Define the y-axis label. If units are available, plot
                if 'units' in ydata.attrs:
                    ztitle = m_field + ''.join([' (', mdata.attrs['units'], ')'])
                else:
                    ztitle = m_field
            # Plot the colorbar
            cbar = plt.colorbar(sc)
            cbar.ax.set_ylabel(ztitle)

        ##if ratio_line is True:
        ##    self.axes[subplot_index].plot(xdata, xdata, 'k--')

        # Define the axe title, x-axis label, y-axis label
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel(ytitle)
        self.axes[subplot_index].set_xlabel(xtitle)

        return self.axes[subplot_index]
