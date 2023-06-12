""" Module for Plotting the Comparison of Datastreams and Variables. """

import matplotlib.pyplot as plt
import numpy as np

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

    def __init__(self, ds, subplot_shape=(1, ), ds_name=None, **kwargs):
        super().__init__(ds, subplot_shape, ds_name, **kwargs)

    def set_xrng(self, xrng, subplot_index=(0, )):
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

    def set_yrng(self, yrng, subplot_index=(0, )):
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

    def set_ratio_line(self, subplot_index=(0, )):
        """
        Sets the 1:1 ratio line.

        Parameters
        ----------
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
        dsname=None,
        cbar_label=None,
        set_title=None,
        subplot_index=(0,),
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
        cbar_label : str
            The desired name to plot for the colorbar
        set_title : str
            The desired title for the plot.
            Default title is created from the datastream.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        subplot_index : tuple
            The subplot index to place the plot in

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

        # Display the scatter plot, pass keyword args for unspecified attributes
        scc = self.axes[subplot_index].scatter(xdata,
                                               ydata,
                                               c=mdata,
                                               **kwargs
                                               )

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
            else:
                ztitle = cbar_label
            # Plot the colorbar
            cbar = plt.colorbar(scc)
            cbar.ax.set_ylabel(ztitle)

        # Define the axe title, x-axis label, y-axis label
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel(ytitle)
        self.axes[subplot_index].set_xlabel(xtitle)

        return self.axes[subplot_index]

    def violin(self,
               field,
               positions=None,
               dsname=None,
               vert=True,
               showmeans=True,
               showmedians=True,
               showextrema=True,
               subplot_index=(0,),
               set_title=None,
               **kwargs,
               ):
        """
        This procedure will produce a violin plot for the selected
        field (or fields).

        Parameters
        ----------
        field : str or list
            The name of the field (or fields) to display on the X axis.
        positions : array-like, Default: None
            The positions of the ticks along dependent axis.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        vert : Boolean, Default: True
            Display violin plot vertical. False will display horizontal.
        showmeans : Boolean; Default: False
            If True, will display the mean of the datastream.
        showmedians : Boolean; Default: False
            If True, will display the medium of the datastream.
        showextrema: Boolean; Default: False
            If True, will display the extremes of the datastream.
        subplot_index : tuple
            The subplot index to place the plot in
        set_title : str
            The title of the plot.

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.violinplot`.

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
        if dsname is None:
            dsname = list(self._ds.keys())[0]

        ds = self._get_data(dsname, field)
        ndata = ds[field]

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        # Define the axes for the figure
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        # Define the axe label. If units are avaiable, plot.
        if 'units' in ndata.attrs:
            axtitle = field + ''.join([' (', ndata.attrs['units'], ')'])
        else:
            axtitle = field

        # Display the scatter plot, pass keyword args for unspecified attributes
        scc = self.axes[subplot_index].violinplot(ndata,
                                                  positions=positions,
                                                  vert=vert,
                                                  showmeans=showmeans,
                                                  showmedians=showmedians,
                                                  showextrema=showextrema,
                                                  **kwargs
                                                  )
        if showmeans is True:
            scc['cmeans'].set_edgecolor('red')
            scc['cmeans'].set_label('mean')
        if showmedians is True:
            scc['cmedians'].set_edgecolor('black')
            scc['cmedians'].set_label('median')
        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )

        # Define the axe title, x-axis label, y-axis label
        self.axes[subplot_index].set_title(set_title)
        if vert is True:
            self.axes[subplot_index].set_ylabel(axtitle)
            if positions is None:
                self.axes[subplot_index].set_xticks([])
        else:
            self.axes[subplot_index].set_xlabel(axtitle)
            if positions is None:
                self.axes[subplot_index].set_yticks([])

        return self.axes[subplot_index]
