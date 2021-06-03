""" Module for Histogram Plotting. """

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .plot import Display
from ..utils import datetime_utils as dt_utils


class HistogramDisplay(Display):
    """
    This class is used to make histogram plots. It is inherited from Display
    and therefore contains all of Display's attributes and methods.

    Examples
    --------
    To create a TimeSeriesDisplay with 3 rows, simply do:

    .. code-block:: python

        ds = act.read_netcdf(the_file)
        disp = act.plotting.HistogramDisplay(
           ds, subplot_shape=(3,), figsize=(15,5))

    The HistogramDisplay constructor takes in the same keyword arguments as
    plt.subplots. For more information on the plt.subplots keyword arguments,
    see the `matplotlib documentation
    <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html>`_.
    If no subplot_shape is provided, then no figure or axis will be created
    until add_subplots or plots is called.

    """
    def __init__(self, obj, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(obj, subplot_shape, ds_name, **kwargs)

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
            raise RuntimeError("set_xrng requires the plot to be displayed.")

        if not hasattr(self, 'xrng') and len(self.axes.shape) == 2:
            self.xrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2),
                                 dtype='datetime64[D]')
        elif not hasattr(self, 'xrng') and len(self.axes.shape) == 1:
            self.xrng = np.zeros((self.axes.shape[0], 2),
                                 dtype='datetime64[D]')

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
            raise RuntimeError("set_yrng requires the plot to be displayed.")

        if not hasattr(self, 'yrng') and len(self.axes.shape) == 2:
            self.yrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2))
        elif not hasattr(self, 'yrng') and len(self.axes.shape) == 1:
            self.yrng = np.zeros((self.axes.shape[0], 2))

        if yrng[0] == yrng[1]:
            yrng[1] = yrng[1] + 1

        self.axes[subplot_index].set_ylim(yrng)
        self.yrng[subplot_index, :] = yrng

    def plot_stacked_bar_graph(self, field, dsname=None, bins=None,
                               sortby_field=None, sortby_bins=None,
                               subplot_index=(0, ), set_title=None,
                               density=False, **kwargs):
        """
        This procedure will plot a stacked bar graph of a histogram.

        Parameters
        ----------
        field : str
            The name of the field to take the histogram of.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        bins : array-like or None
            The histogram bin boundaries to use. Set to None to use
            numpy's default boundaries.
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
        density: bool
            Set to True to plot a p.d.f. instead of a frequency histogram.

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.bar`.

        Returns
        -------
        return_dict : dict
            A dictionary containing the plot axis handle, bin boundaries, and
            generated histogram.

        """
        if dsname is None and len(self._obj.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 " +
                              "or more datasets in the TimeSeriesDisplay " +
                              "object."))
        elif dsname is None:
            dsname = list(self._obj.keys())[0]

        xdata = self._obj[dsname][field]

        if 'units' in xdata.attrs:
            xtitle = ''.join(['(', xdata.attrs['units'], ')'])
        else:
            xtitle = field

        if sortby_field is not None:
            ydata = self._obj[dsname][sortby_field]

        if bins is not None and sortby_bins is None and sortby_field is not None:
            # We will defaut the y direction to have the same # of bins as x
            sortby_bins = np.linspace(ydata.values.min(), ydata.values.max(), len(bins))

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
            if bins is None:
                my_hist, x_bins, y_bins = np.histogram2d(
                    xdata.values.flatten(), ydata.values.flatten(), density=density)
            else:
                my_hist, x_bins, y_bins = np.histogram2d(
                    xdata.values.flatten(), ydata.values.flatten(),
                    density=density, bins=[bins, sortby_bins])
            x_inds = (x_bins[:-1] + x_bins[1:]) / 2.0
            self.axes[subplot_index].bar(
                x_inds, my_hist[:, 0].flatten(),
                label=(str(y_bins[0]) + " to " + str(y_bins[1])), **kwargs)
            for i in range(1, len(y_bins) - 1):
                self.axes[subplot_index].bar(
                    x_inds, my_hist[:, i].flatten(),
                    bottom=my_hist[:, i - 1],
                    label=(str(y_bins[i]) + " to " + str(y_bins[i + 1])), **kwargs)
            self.axes[subplot_index].legend()
        else:
            if bins is None:
                bmin = np.nanmin(xdata)
                bmax = np.nanmax(xdata)
                bins = np.arange(bmin, bmax, (bmax - bmin) / 10.)
            my_hist, bins = np.histogram(
                xdata.values.flatten(), bins=bins, density=density)
            x_inds = (bins[:-1] + bins[1:]) / 2.0
            self.axes[subplot_index].bar(x_inds, my_hist)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, field, 'on',
                                  dt_utils.numpy_to_arm_date(
                                      self._obj[dsname].time.values[0])])
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel("count")
        self.axes[subplot_index].set_xlabel(xtitle)

        return_dict = {}
        return_dict["plot_handle"] = self.axes[subplot_index]
        if 'x_bins' in locals():
            return_dict["x_bins"] = x_bins
            return_dict["y_bins"] = y_bins
        else:
            return_dict["bins"] = bins
        return_dict["histogram"] = my_hist

        return return_dict

    def plot_size_distribution(self, field, bins, time=None, dsname=None,
                               subplot_index=(0, ), set_title=None,
                               **kwargs):
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
        if dsname is None and len(self._obj.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 " +
                              "or more datasets in the TimeSeriesDisplay " +
                              "object."))
        elif dsname is None:
            dsname = list(self._obj.keys())[0]

        xdata = self._obj[dsname][field]

        if isinstance(bins, str):
            bins = self._obj[dsname][bins]
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

        if(len(xdata.dims) > 1 and time is None):
            raise ValueError(("Input data has more than one dimension, " +
                              "you must specify a time to plot!"))
        elif len(xdata.dims) > 1:
            xdata = xdata.sel(time=time, method='nearest')

        if(len(bins.dims) > 1 or len(bins.values) != len(xdata.values)):
            raise ValueError("Bins must be a one dimensional field whose " +
                             "length is equal to the field length!")

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, field, 'on',
                                  dt_utils.numpy_to_arm_date(
                                      self._obj[dsname].time.values[0])])

        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].step(bins.values, xdata.values)
        self.axes[subplot_index].set_xlabel(xtitle)
        self.axes[subplot_index].set_ylabel(ytitle)

        return self.axes[subplot_index]

    def plot_stairstep_graph(self, field, dsname=None, bins=None,
                             sortby_field=None, sortby_bins=None,
                             subplot_index=(0, ),
                             set_title=None,
                             density=False, **kwargs):
        """
        This procedure will plot a stairstep plot of a histogram.

        Parameters
        ----------
        field : str
            The name of the field to take the histogram of.
        dsname : str or None
            The name of the datastream the field is contained in. Set
            to None to let ACT automatically determine this.
        bins : array-like or None
            The histogram bin boundaries to use. Set to None to use
            numpy's default boundaries.
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

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.step`.

        Returns
        -------
        return_dict : dict
             A dictionary containing the plot axis handle, bin boundaries, and
             generated histogram.

        """
        if dsname is None and len(self._obj.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 " +
                              "or more datasets in the TimeSeriesDisplay " +
                              "object."))
        elif dsname is None:
            dsname = list(self._obj.keys())[0]

        xdata = self._obj[dsname][field]

        if 'units' in xdata.attrs:
            xtitle = ''.join(['(', xdata.attrs['units'], ')'])
        else:
            xtitle = field

        if sortby_field is not None:
            ydata = self._obj[dsname][sortby_field]

        if bins is not None and sortby_bins is None and sortby_field is not None:
            # We will defaut the y direction to have the same # of bins as x
            sortby_bins = np.linspace(ydata.values.min(), ydata.values.max(), len(bins))

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
            if bins is None:
                my_hist, x_bins, y_bins = np.histogram2d(
                    xdata.values.flatten(), ydata.values.flatten(), density=density)
            else:
                my_hist, x_bins, y_bins = np.histogram2d(
                    xdata.values.flatten(), ydata.values.flatten(),
                    density=density, bins=[bins, sortby_bins])
            x_inds = (x_bins[:-1] + x_bins[1:]) / 2.0
            self.axes[subplot_index].step(
                x_inds, my_hist[:, 0].flatten(),
                label=(str(y_bins[0]) + " to " + str(y_bins[1])), **kwargs)
            for i in range(1, len(y_bins) - 1):
                self.axes[subplot_index].step(
                    x_inds, my_hist[:, i].flatten(),
                    label=(str(y_bins[i]) + " to " + str(y_bins[i + 1])), **kwargs)
            self.axes[subplot_index].legend()
        else:
            my_hist, bins = np.histogram(
                xdata.values.flatten(), bins=bins, density=density)
            x_inds = (bins[:-1] + bins[1:]) / 2.0
            self.axes[subplot_index].step(x_inds, my_hist, **kwargs)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, field, 'on',
                                  dt_utils.numpy_to_arm_date(
                                      self._obj[dsname].time.values[0])])
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel("count")
        self.axes[subplot_index].set_xlabel(xtitle)

        return_dict = {}
        return_dict["plot_handle"] = self.axes[subplot_index]
        if 'x_bins' in locals():
            return_dict["x_bins"] = x_bins
            return_dict["y_bins"] = y_bins
        else:
            return_dict["bins"] = bins
        return_dict["histogram"] = my_hist

        return return_dict

    def plot_heatmap(self, x_field, y_field, dsname=None, x_bins=None, y_bins=None,
                     subplot_index=(0, ), set_title=None,
                     density=False, **kwargs):
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
        x_bins : array-like or None
            The histogram bin boundaries to use for the variable on the X axis.
            Set to None to use numpy's default boundaries.
        y_bins : array-like or None
            The histogram bin boundaries to use for the variable on the Y axis.
            Set to None to use numpy's default boundaries.
        subplot_index : tuple
            The subplot index to place the plot in
        set_title : str
            The title of the plot.
        density : bool
            Set to True to plot a p.d.f. instead of a frequency histogram.

        Other keyword arguments will be passed into :func:`matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        return_dict : dict
            A dictionary containing the plot axis handle, bin boundaries, and
            generated histogram.

        """
        if dsname is None and len(self._obj.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._obj.keys())[0]

        xdata = self._obj[dsname][x_field]

        if 'units' in xdata.attrs:
            xtitle = ''.join(['(', xdata.attrs['units'], ')'])
        else:
            xtitle = x_field
        ydata = self._obj[dsname][y_field]

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
                xdata.values.flatten(), ydata.values.flatten(), density=density)
        else:
            my_hist, x_bins, y_bins = np.histogram2d(
                xdata.values.flatten(), ydata.values.flatten(),
                density=density, bins=[x_bins, y_bins])
        x_inds = (x_bins[:-1] + x_bins[1:]) / 2.0
        y_inds = (y_bins[:-1] + y_bins[1:]) / 2.0
        xi, yi = np.meshgrid(x_inds, y_inds, indexing='ij')
        mesh = self.axes[subplot_index].pcolormesh(xi, yi, my_hist, **kwargs)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, 'on',
                                  dt_utils.numpy_to_arm_date(
                                      self._obj[dsname].time.values[0])])
        self.axes[subplot_index].set_title(set_title)
        self.axes[subplot_index].set_ylabel(ytitle)
        self.axes[subplot_index].set_xlabel(xtitle)
        self.add_colorbar(mesh, title="count", subplot_index=subplot_index)

        return_dict = {}
        return_dict["plot_handle"] = self.axes[subplot_index]
        return_dict["x_bins"] = x_bins
        return_dict["y_bins"] = y_bins
        return_dict["histogram"] = my_hist

        return return_dict
