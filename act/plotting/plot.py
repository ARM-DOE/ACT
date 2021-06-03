"""
Class for creating timeseries plots from ACT datasets.

"""

# Import third party libraries
import matplotlib.pyplot as plt
import numpy as np
import warnings
import xarray as xr


class Display(object):
    """
    This class is the base class for all of the other Display object
    types in ACT. This contains the common attributes and routines
    between the differing *Display* classes. We recommend that you
    use the classes inherited from Display for making your plots
    such as :func:`act.plotting.TimeSeriesDisplay` and
    :func:`act.plotting.WindRoseDisplay` instead of
    trying to do so using the Display object.

    However, we do ask that if you add another object to the plotting
    module of ACT that you make it a subclass of Display. Display provides
    some basic functionality for the handling of datasets and subplot
    parameters.

    Attributes
    ----------
    fields : dict
        The dictionary containing the fields inside the ARM dataset. Each field
        has a key that links to an xarray DataArray object.
    ds : str
        The name of the datastream.
    file_dates : list
        The dates of each file being displayed.
    fig : matplotlib figure handle
        The matplotlib figure handle to display the plots on. Initializing the
        class with this set to None will create a new figure handle. See the
        matplotlib documentation on what keyword arguments are
        available.
    axes : list
        The list of axes handles to each subplot.
    plot_vars : list
        The list of variables being plotted.
    cbs : list
        The list of colorbar handles.

    Parameters
    ----------
    obj : ACT Dataset, dict, or tuple
        The ACT Dataset to display in the object. If more than one dataset
        is to be specified, then a tuple can be used if all of the datasets
        conform to ARM standards. Otherwise, a dict with a key corresponding
        to the name of each datastream will need to be supplied in order
        to create the ability to plot multiple datasets.
    subplot_shape : 1 or 2D tuple
        A tuple representing the number of (rows, columns) for the subplots
        in the display. If this is None, the figure and axes will not
        be initialized.
    ds_name : str or None
        The name of the datastream to plot. This is only used if a non-ARM
        compliant dataset is being loaded and if only one such dataset is
        loaded.
    subplot_kw : dict, optional
        The kwargs to pass into :func:`fig.subplots`
    **kwargs : keywords arguments
        Keyword arguments passed to :func:`plt.figure`.

    """
    def __init__(self, obj, subplot_shape=(1,), ds_name=None,
                 subplot_kw=None, **kwargs):
        if isinstance(obj, xr.Dataset):
            if 'datastream' in obj.attrs.keys() is not None:
                self._obj = {obj.attrs['datastream']: obj}
            elif ds_name is not None:
                self._obj = {ds_name: obj}
            else:
                warnings.warn(("Could not discern datastream" +
                               "name and dict or tuple were " +
                               "not provided. Using default" +
                               "name of act_datastream!"), UserWarning)

                self._obj = {'act_datastream': obj}

        # Automatically name by datastream if a tuple of object is supplied
        if isinstance(obj, tuple):
            self._obj = {}
            for objs in obj:
                self._obj[objs.attrs['datastream']] = objs

        if isinstance(obj, dict):
            self._obj = obj

        self.fields = {}
        self.ds = {}
        self.file_dates = {}
        self.xrng = np.zeros((1, 2))
        self.yrng = np.zeros((1, 2))

        for dsname in self._obj.keys():
            self.fields[dsname] = self._obj[dsname].variables
            if '_datastream' in self._obj[dsname].attrs.keys():
                self.ds[dsname] = str(self._obj[dsname].attrs['_datastream'])
            else:
                self.ds[dsname] = "act_datastream"
            if '_file_dates' in self._obj[dsname].attrs.keys():
                self.file_dates[dsname] = self._obj[dsname].attrs['_file_dates']

        self.fig = None
        self.axes = None
        self.plot_vars = []
        self.cbs = []
        if subplot_shape is not None:
            self.add_subplots(subplot_shape, subplot_kw=subplot_kw,
                              **kwargs)

    def add_subplots(self, subplot_shape=(1,), subplot_kw=None,
                     **kwargs):
        """
        Adds subplots to the Display object. The current
        figure in the object will be deleted and overwritten.

        Parameters
        ----------
        subplot_shape : 1 or 2D tuple, list, or array
            The structure of the subplots in (rows, cols).
        subplot_kw : dict, optional
            The kwargs to pass into fig.subplots.
        **kwargs : keyword arguments
            Any other keyword arguments that will be passed
            into :func:`matplotlib.pyplot.subplots`. See the matplotlib
            documentation for further details on what keyword
            arguments are available.

        """
        if self.fig is not None:
            plt.close(self.fig)
            del self.fig

        if len(subplot_shape) == 2:
            fig, ax = plt.subplots(
                subplot_shape[0], subplot_shape[1],
                subplot_kw=subplot_kw,
                **kwargs)
            self.xrng = np.zeros((subplot_shape[0], subplot_shape[1], 2))
            self.yrng = np.zeros((subplot_shape[0], subplot_shape[1], 2))
            if subplot_shape[0] == 1:
                ax = ax.reshape(1, subplot_shape[1])
        elif len(subplot_shape) == 1:
            fig, ax = plt.subplots(
                subplot_shape[0], 1, subplot_kw=subplot_kw, **kwargs)
            if subplot_shape[0] == 1:
                ax = np.array([ax])
            self.xrng = np.zeros((subplot_shape[0], 2))
            self.yrng = np.zeros((subplot_shape[0], 2))
        else:
            raise ValueError(("subplot_shape must be a 1 or 2 dimensional" +
                              "tuple list, or array!"))
        self.fig = fig
        self.axes = ax

    def put_display_in_subplot(self, display, subplot_index):
        """
        This will place a Display object into a specific subplot.
        The display object must only have one subplot.

        This will clear the display in the Display object being added.

        Parameters
        ----------
        Display : Display object or subclass
            The Display object to add as a subplot
        subplot_index : tuple
            Which subplot to add the Display to.

        Returns
        -------
        ax : matplotlib axis handle
            The axis handle to the display object being added.

        """
        if len(display.axes) > 1:
            raise RuntimeError("Only single plots can be made as subplots " +
                               "of another Display object!")

        my_projection = display.axes[0].name
        plt.close(display.fig)
        display.fig = self.fig
        self.fig.delaxes(self.axes[subplot_index])
        the_shape = self.axes.shape
        if len(the_shape) == 1:
            second_value = 1
        else:
            second_value = the_shape[1]

        self.axes[subplot_index] = self.fig.add_subplot(
            the_shape[0], second_value,
            (second_value - 1) * the_shape[0] + subplot_index[0] + 1,
            projection=my_projection)

        display.axes = np.array([self.axes[subplot_index]])

        return display.axes[0]

    def assign_to_figure_axis(self, fig, ax):
        """
        This assigns the Display to a specific figure and axis.
        This will remove the figure and axes that are currently
        stored in the object. The display object will then only
        have one axis handle.

        Parameters
        ----------
        fig : matplotlib figure handle
            The figure to place the time series display in.
        ax : axis handle
            The axis handle to place the plot in.

        """
        if self.fig is not None:
            plt.close(self.fig)
            del self.fig

        del self.axes
        self.fig = fig
        self.axes = np.array([ax])

    def add_colorbar(self, mappable, title=None, subplot_index=(0, )):
        """
        Adds a colorbar to the plot.

        Parameters
        ----------
        mappable : matplotlib mappable
            The mappable to base the colorbar on.
        title : str
            The title of the colorbar. Set to None to have no title.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.

        Returns
        -------
        cbar : matplotlib colorbar handle
            The handle to the matplotlib colorbar.

        """
        if self.axes is None:
            raise RuntimeError("add_colorbar requires the plot "
                               "to be displayed.")

        fig = self.fig
        ax = self.axes[subplot_index]

        # Give the colorbar it's own axis so the 2D plots line up with 1D
        box = ax.get_position()
        pad, width = 0.01, 0.01
        cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
        cbar = plt.colorbar(mappable, cax=cax)
        cbar.ax.set_ylabel(title, rotation=270, fontsize=8, labelpad=3)
        cbar.ax.tick_params(labelsize=6)
        self.cbs.append(cbar)

        return cbar
