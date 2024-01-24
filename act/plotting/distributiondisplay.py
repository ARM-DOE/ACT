""" Module for Distribution Plotting. """

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

from ..utils import datetime_utils as dt_utils
from .plot import Display


class DistributionDisplay(Display):
    """
    This class is used to make  distribution related plots. It is inherited from Display
    and therefore contains all of Display's attributes and methods.

    Examples
    --------
    To create a DistributionDisplay with 3 rows, simply do:

    .. code-block:: python

        ds = act.io.read_arm_netcdf(the_file)
        disp = act.plotting.DistsributionDisplay(ds, subplot_shape=(3,), figsize=(15, 5))

    The DistributionDisplay constructor takes in the same keyword arguments as
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

    def _get_data(self, dsname, fields):
        if isinstance(fields, str):
            fields = [fields]
        return self._ds[dsname][fields].dropna('time')

    def plot_stacked_bar(
        self,
        field,
        dsname=None,
        bins=10,
        sortby_field=None,
        sortby_bins=None,
        subplot_index=(0,),
        set_title=None,
        density=False,
        hist_kwargs=dict(),
        **kwargs,
    ):
        """
        This procedure will plot a stacked bar graph of a histogram.

        Parameters
        ----------
        field : str
            The name of the field to take the histogram of.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        bins : array-like or int
            The histogram bin boundaries to use. If not specified, numpy's
            default 10 is used.
        sortby_field : str or None
            Set this option to a field name in order to sort the histograms
            by a given field parameter. For example, one can sort histograms of CO2
            concentration by temperature.
        sortby_bins : array-like or None
            The bins to sort the histograms by.
        subplot_index : tuple
            The subplot index to place the plot in
        set_title : str
            The title of the plot.
        density : bool
            Set to True to plot a p.d.f. instead of a frequency histogram.
        hist_kwargs : dict
            Additional keyword arguments to pass to numpy histogram.

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.bar`.

        Returns
        -------
        return_dict : dict
            A dictionary containing the plot axis handle, bin boundaries, and
            generated histogram.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                + 'or more datasets in the TimeSeriesDisplay '
                + 'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        if sortby_field is not None:
            ds = self._get_data(dsname, [field, sortby_field])
            xdata, ydata = ds[field], ds[sortby_field]
        else:
            xdata = self._get_data(dsname, field)[field]

        if 'units' in xdata.attrs:
            xtitle = ''.join(['(', xdata.attrs['units'], ')'])
        else:
            xtitle = field

        if sortby_bins is None and sortby_field is not None:
            # We will defaut the y direction to have the same # of bins as x
            if isinstance(bins, int):
                n_bins = bins
            else:
                n_bins = len(bins)
            sortby_bins = np.linspace(ydata.values.min(), ydata.values.max(), n_bins)

        # Get the current plotting axis
        if self.fig is None:
            self.fig = plt.figure()
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        if sortby_field is not None:
            if 'units' in ydata.attrs:
                ytitle = ''.join(['(', ydata.attrs['units'], ')'])
            else:
                ytitle = field
            my_hist, x_bins, y_bins = np.histogram2d(
                xdata.values.flatten(),
                ydata.values.flatten(),
                density=density,
                bins=[bins, sortby_bins],
                **hist_kwargs,
            )
            x_inds = (x_bins[:-1] + x_bins[1:]) / 2.0
            self.axes[subplot_index].bar(
                x_inds,
                my_hist[:, 0].flatten(),
                label=(str(y_bins[0]) + ' to ' + str(y_bins[1])),
                **kwargs,
            )
            for i in range(1, len(y_bins) - 1):
                self.axes[subplot_index].bar(
                    x_inds,
                    my_hist[:, i].flatten(),
                    bottom=my_hist[:, i - 1],
                    label=(str(y_bins[i]) + ' to ' + str(y_bins[i + 1])),
                    **kwargs,
                )
            self.axes[subplot_index].legend()
        else:
            my_hist, bins = np.histogram(
                xdata.values.flatten(), bins=bins, density=density, **hist_kwargs
            )
            x_inds = (bins[:-1] + bins[1:]) / 2.0
            self.axes[subplot_index].bar(x_inds, my_hist)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    field,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel('count')
        self.axes[subplot_index].set_xlabel(xtitle)

        return_dict = {}
        return_dict['plot_handle'] = self.axes[subplot_index]
        if 'x_bins' in locals():
            return_dict['x_bins'] = x_bins
            return_dict['y_bins'] = y_bins
        else:
            return_dict['bins'] = bins
        return_dict['histogram'] = my_hist

        return return_dict

    def plot_size_distribution(
        self, field, bins, time=None, dsname=None, subplot_index=(0,), set_title=None, **kwargs
    ):
        """
        This procedure plots a stairstep plot of a size distribution. This is
        useful for plotting size distributions and waveforms.

        Parameters
        ----------
        field : str
            The name of the field to plot the spectrum from.
        bins : str or array-like
            The name of the field that stores the bins for the spectra.
        time : none or datetime
            If None, spectra to plot will be automatically determined.
            Otherwise, specify this field for the time period to plot.
        dsname : str
            The name of the Dataset to plot. Set to None to have
            ACT automatically determine this.
        subplot_index : tuple
            The subplot index to place the plot in.
        set_title : str or None
            Use this to set the title.

        Additional keyword arguments will be passed into :func:`matplotlib.pyplot.step`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle referring to the plot.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                + 'or more datasets in the TimeSeriesDisplay '
                + 'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        xdata = self._get_data(dsname, field)[field]

        if isinstance(bins, str):
            bins = self._ds[dsname][bins]
        else:
            bins = xr.DataArray(bins)

        if 'units' in bins.attrs:
            xtitle = ''.join(['(', bins.attrs['units'], ')'])
        else:
            xtitle = 'Bin #'

        if 'units' in xdata.attrs:
            ytitle = ''.join(['(', xdata.attrs['units'], ')'])
        else:
            ytitle = field

        if len(xdata.dims) > 1 and time is None:
            raise ValueError(
                'Input data has more than one dimension, ' + 'you must specify a time to plot!'
            )
        elif len(xdata.dims) > 1:
            xdata = xdata.sel(time=time, method='nearest')

        if len(bins.dims) > 1 or len(bins.values) != len(xdata.values):
            raise ValueError(
                'Bins must be a one dimensional field whose '
                + 'length is equal to the field length!'
            )

        # Get the current plotting axis
        if self.fig is None:
            self.fig = plt.figure()
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    field,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )
            if time is not None:
                t = pd.Timestamp(time)
                set_title += ''.join(
                    [' at ', ':'.join([str(t.hour), str(t.minute), str(t.second)])]
                )
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].step(bins.values, xdata.values, **kwargs)
        self.axes[subplot_index].set_xlabel(xtitle)
        self.axes[subplot_index].set_ylabel(ytitle)

        return self.axes[subplot_index]

    def plot_stairstep(
        self,
        field,
        dsname=None,
        bins=10,
        sortby_field=None,
        sortby_bins=None,
        subplot_index=(0,),
        set_title=None,
        density=False,
        hist_kwargs=dict(),
        **kwargs,
    ):
        """
        This procedure will plot a stairstep plot of a histogram.

        Parameters
        ----------
        field : str
            The name of the field to take the histogram of.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        bins : array-like or int
            The histogram bin boundaries to use. If not specified, numpy's
            default 10 is used.
        sortby_field : str or None
            Set this option to a field name in order to sort the histograms
            by a given field parameter. For example, one can sort histograms of CO2
            concentration by temperature.
        sortby_bins : array-like or None
            The bins to sort the histograms by.
        subplot_index : tuple
            The subplot index to place the plot in.
        set_title : str
            The title of the plot.
        density : bool
            Set to True to plot a p.d.f. instead of a frequency histogram.
        hist_kwargs : dict
            Additional keyword arguments to pass to numpy histogram.

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.step`.

        Returns
        -------
        return_dict : dict
             A dictionary containing the plot axis handle, bin boundaries, and
             generated histogram.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                + 'or more datasets in the TimeSeriesDisplay '
                + 'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        xdata = self._get_data(dsname, field)[field]

        if 'units' in xdata.attrs:
            xtitle = ''.join(['(', xdata.attrs['units'], ')'])
        else:
            xtitle = field

        if sortby_field is not None:
            ydata = self._ds[dsname][sortby_field]

        if sortby_bins is None and sortby_field is not None:
            if isinstance(bins, int):
                n_bins = bins
            else:
                n_bins = len(bins)
            # We will defaut the y direction to have the same # of bins as x
            sortby_bins = np.linspace(ydata.values.min(), ydata.values.max(), n_bins)

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        if sortby_field is not None:
            if 'units' in ydata.attrs:
                ytitle = ''.join(['(', ydata.attrs['units'], ')'])
            else:
                ytitle = field
            my_hist, x_bins, y_bins = np.histogram2d(
                xdata.values.flatten(),
                ydata.values.flatten(),
                density=density,
                bins=[bins, sortby_bins],
                **hist_kwargs,
            )
            x_inds = (x_bins[:-1] + x_bins[1:]) / 2.0
            self.axes[subplot_index].step(
                x_inds,
                my_hist[:, 0].flatten(),
                label=(str(y_bins[0]) + ' to ' + str(y_bins[1])),
                **kwargs,
            )
            for i in range(1, len(y_bins) - 1):
                self.axes[subplot_index].step(
                    x_inds,
                    my_hist[:, i].flatten(),
                    label=(str(y_bins[i]) + ' to ' + str(y_bins[i + 1])),
                    **kwargs,
                )
            self.axes[subplot_index].legend()
        else:
            my_hist, bins = np.histogram(
                xdata.values.flatten(), bins=bins, density=density, **hist_kwargs
            )

            x_inds = (bins[:-1] + bins[1:]) / 2.0
            self.axes[subplot_index].step(x_inds, my_hist, **kwargs)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    field,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel('count')
        self.axes[subplot_index].set_xlabel(xtitle)

        return_dict = {}
        return_dict['plot_handle'] = self.axes[subplot_index]
        if 'x_bins' in locals():
            return_dict['x_bins'] = x_bins
            return_dict['y_bins'] = y_bins
        else:
            return_dict['bins'] = bins
        return_dict['histogram'] = my_hist

        return return_dict

    def plot_heatmap(
        self,
        x_field,
        y_field,
        dsname=None,
        x_bins=None,
        y_bins=None,
        subplot_index=(0,),
        set_title=None,
        density=False,
        set_shading='auto',
        hist_kwargs=dict(),
        threshold=None,
        **kwargs,
    ):
        """
        This procedure will plot a heatmap of a histogram from 2 variables.

        Parameters
        ----------
        x_field : str
            The name of the field to take the histogram of on the X axis.
        y_field : str
            The name of the field to take the histogram of on the Y axis.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        x_bins : array-like, int, or None
            The histogram bin boundaries to use for the variable on the X axis.
            Set to None to use numpy's default boundaries.
            If an int, will indicate the number of bins to use
        y_bins : array-like, int, or None
            The histogram bin boundaries to use for the variable on the Y axis.
            Set to None to use numpy's default boundaries.
            If an int, will indicate the number of bins to use
        subplot_index : tuple
            The subplot index to place the plot in
        set_title : str
            The title of the plot.
        density : bool
            Set to True to plot a p.d.f. instead of a frequency histogram.
        set_shading : string
            Option to to set the matplotlib.pcolormesh shading parameter.
            Default to 'auto'
        threshold : float
            Value on which to threshold the histogram results for plotting.
            Setting to 0 will ensure that all 0 values are removed from the plot
            making it easier to distringuish between 0 and low values
        hist_kwargs : Additional keyword arguments to pass to numpy histogram.

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        return_dict : dict
            A dictionary containing the plot axis handle, bin boundaries, and
            generated histogram.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        ds = self._get_data(dsname, [x_field, y_field])
        xdata, ydata = ds[x_field], ds[y_field]

        if 'units' in xdata.attrs:
            xtitle = ''.join(['(', xdata.attrs['units'], ')'])
        else:
            xtitle = x_field

        if x_bins is not None and isinstance(x_bins, int):
            x_bins = np.linspace(xdata.values.min(), xdata.values.max(), x_bins)

        if y_bins is not None and isinstance(x_bins, int):
            y_bins = np.linspace(ydata.values.min(), ydata.values.max(), y_bins)

        if x_bins is not None and y_bins is None:
            # We will defaut the y direction to have the same # of bins as x
            y_bins = np.linspace(ydata.values.min(), ydata.values.max(), len(x_bins))

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        if 'units' in ydata.attrs:
            ytitle = ''.join(['(', ydata.attrs['units'], ')'])
        else:
            ytitle = y_field

        if x_bins is None:
            my_hist, x_bins, y_bins = np.histogram2d(
                xdata.values.flatten(), ydata.values.flatten(), density=density, **hist_kwargs
            )
        else:
            my_hist, x_bins, y_bins = np.histogram2d(
                xdata.values.flatten(),
                ydata.values.flatten(),
                density=density,
                bins=[x_bins, y_bins],
                **hist_kwargs,
            )
        # Adding in the ability to threshold the heatmaps
        if threshold is not None:
            my_hist[my_hist <= threshold] = np.nan

        x_inds = (x_bins[:-1] + x_bins[1:]) / 2.0
        y_inds = (y_bins[:-1] + y_bins[1:]) / 2.0
        xi, yi = np.meshgrid(x_inds, y_inds, indexing='ij')
        mesh = self.axes[subplot_index].pcolormesh(xi, yi, my_hist, shading=set_shading, **kwargs)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel(ytitle)
        self.axes[subplot_index].set_xlabel(xtitle)
        self.add_colorbar(mesh, title='count', subplot_index=subplot_index)

        return_dict = {}
        return_dict['plot_handle'] = self.axes[subplot_index]
        return_dict['x_bins'] = x_bins
        return_dict['y_bins'] = y_bins
        return_dict['histogram'] = my_hist

        return return_dict

    def set_ratio_line(self, subplot_index=(0,)):
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
        ratio = np.linspace(xlims, xlims[-1])
        self.axes[subplot_index].plot(ratio, ratio, 'k--')

    def plot_scatter(
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
        scc = self.axes[subplot_index].scatter(xdata, ydata, c=mdata, **kwargs)

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
                if 'units' in mdata.attrs:
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

    def plot_violin(
        self,
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
        scc = self.axes[subplot_index].violinplot(
            ndata,
            positions=positions,
            vert=vert,
            showmeans=showmeans,
            showmedians=showmedians,
            showextrema=showextrema,
            **kwargs,
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
