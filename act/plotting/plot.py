"""
Class for creating timeseries plots from ACT datasets.

"""

import warnings

# Import third party libraries
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import inspect


class Display:
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
    ds : ACT xarray.Dataset, dict, or tuple
        The ACT xarray dataset to display in the object. If more than one dataset
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

    def __init__(self, ds, subplot_shape=(1,), ds_name=None, subplot_kw=None, **kwargs):
        if isinstance(ds, xr.Dataset):
            if 'datastream' in ds.attrs.keys() is not None:
                self._ds = {ds.attrs['datastream']: ds}
            elif ds_name is not None:
                self._ds = {ds_name: ds}
            else:
                warnings.warn(
                    (
                        'Could not discern datastream'
                        + 'name and dict or tuple were '
                        + 'not provided. Using default'
                        + 'name of act_datastream!'
                    ),
                    UserWarning,
                )

                self._ds = {'act_datastream': ds}

        # Automatically name by datastream if a tuple of datasets is supplied
        if isinstance(ds, tuple):
            self._ds = {}
            for multi_ds in ds:
                self._ds[multi_ds.attrs['datastream']] = multi_ds

        if isinstance(ds, dict):
            self._ds = ds

        self.fields = {}
        self.ds = {}
        self.file_dates = {}
        self.xrng = np.zeros((1, 2))
        self.yrng = np.zeros((1, 2))

        for dsname in self._ds.keys():
            self.fields[dsname] = self._ds[dsname].variables
            if '_datastream' in self._ds[dsname].attrs.keys():
                self.ds[dsname] = str(self._ds[dsname].attrs['_datastream'])
            else:
                self.ds[dsname] = 'act_datastream'
            if '_file_dates' in self._ds[dsname].attrs.keys():
                self.file_dates[dsname] = self._ds[dsname].attrs['_file_dates']

        self.fig = None
        self.axes = None
        self.plot_vars = []
        self.cbs = []
        if subplot_shape is not None:
            self.add_subplots(subplot_shape, subplot_kw=subplot_kw, **kwargs)

    def add_subplots(self, subplot_shape=(1,), secondary_y=False, subplot_kw=None, **kwargs):
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
                subplot_shape[0], subplot_shape[1], subplot_kw=subplot_kw, **kwargs
            )
            self.xrng = np.zeros((subplot_shape[0], subplot_shape[1], 2))
            self.yrng = np.zeros((subplot_shape[0], subplot_shape[1], 2))
            if subplot_shape[0] == 1:
                ax = ax.reshape(1, subplot_shape[1])
        elif len(subplot_shape) == 1:
            fig, ax = plt.subplots(subplot_shape[0], 1, subplot_kw=subplot_kw, **kwargs)
            if subplot_shape[0] == 1:
                ax = np.array([ax])
            self.xrng = np.zeros((subplot_shape[0], 2))
            self.yrng = np.zeros((subplot_shape[0], 2))
        else:
            raise ValueError('subplot_shape must be a 1 or 2 dimensional' + 'tuple list, or array!')

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
            raise RuntimeError(
                'Only single plots can be made as subplots ' + 'of another Display object!'
            )
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
            the_shape[0],
            second_value,
            (second_value - 1) * the_shape[0] + subplot_index[0] + 1,
            projection=my_projection,
        )

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

    def add_colorbar(
        self, mappable, title=None, subplot_index=(0,), pad=None, width=None, **kwargs
    ):
        """
        Adds a colorbar to the plot.

        Parameters
        ----------
        mappable : matplotlib mappable
            The mappable to base the colorbar on.
        title : str
            The title of the colorbar. Set to None to have no title.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range
        pad : float
            Padding to right of plot for placement of the colorbar
        width : float
            Width of the colorbar
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.colorbar`

        Returns
        -------
        cbar : matplotlib colorbar handle
            The handle to the matplotlib colorbar.

        """
        if self.axes is None:
            raise RuntimeError('add_colorbar requires the plot ' 'to be displayed.')

        fig = self.fig
        ax = self.axes[subplot_index]

        if pad is None:
            pad = 0.01

        if width is None:
            width = 0.01

        # Give the colorbar it's own axis so the 2D plots line up with 1D
        box = ax.get_position()
        cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
        cbar = plt.colorbar(mappable, cax=cax, **kwargs)
        if title is not None:
            cbar.ax.set_ylabel(title, rotation=270, fontsize=8, labelpad=3)
        cbar.ax.tick_params(labelsize=6)
        self.cbs.append(cbar)

        return cbar

    def group_by(self, units):
        """
        Group the Display by specific units of time.

        Parameters
        ----------
        units: str
            One of: 'year', 'month', 'day', 'hour', 'minute', 'second'.
            Group the plot by this unit of time (year, month, etc.)
        Returns
        -------
        groupby: act.plotting.DisplayGroupby
            The DisplayGroupby object to be retuned.
        """
        return DisplayGroupby(self, units)


class DisplayGroupby:
    def __init__(self, display, units):
        """

        Parameters
        ----------
        display: Display
            The Display object to group by time.
        units: str
            The time units to group by. Can be one of:
            'year', 'month', 'day', 'hour', 'minute', 'second'
        """
        self.display = display
        self._groupby = {}
        self.mapping = {}
        self.xlims = {}
        self.units = units
        self.isTimeSeriesDisplay = hasattr(self.display, 'time_height_scatter')
        num_groups = 0
        datastreams = list(display._ds.keys())
        for key in datastreams:
            self._groupby[key] = display._ds[key].groupby('time.%s' % units)
            num_groups = max([num_groups, len(self._groupby[key])])

    def plot_group(self, func_name, dsname=None, **kwargs):
        """
        Plots each group created in :func:`act.plotting.Display.group_by` into each subplot of the display.
        Parameters
        ----------
        func_name: str
            The name of the plotting function in the Display that you are grouping.
        dsname: str or None
            The name of the datastream to plot

        Additional keyword objects are passed into *func_name*.

        Returns
        -------
        axis: Array of matplotlib axes handles
            The array of matplotlib axes handles that correspond to each subplot.
        """
        if dsname is None:
            dsname = list(self.display._ds.keys())[0].split('_')[0]

        func = getattr(self.display, func_name)

        if not callable(func):
            raise RuntimeError("The specified string is not a function of " "the Display object.")
        subplot_shape = self.display.axes.shape

        i = 0
        wrap_around = False
        old_ds = self.display._ds
        for key in self._groupby.keys():
            if dsname == key:
                self.display._ds = {}
                for k, ds in self._groupby[key]:
                    num_years = len(np.unique(ds.time.dt.year))
                    self.display._ds[key + '_%d' % k] = ds
                    if i >= np.prod(subplot_shape):
                        i = 0
                        wrap_around = True
                    if len(subplot_shape) == 2:
                        subplot_index = (int(i / subplot_shape[1]), i % subplot_shape[1])
                    else:
                        subplot_index = (i % subplot_shape[0],)
                    args, varargs, varkw, _, _, _, _ = inspect.getfullargspec(func)
                    if "subplot_index" in args:
                        kwargs["subplot_index"] = subplot_index
                    if "time_rng" in args:
                        kwargs["time_rng"] = (ds.time.values.min(), ds.time.values.max())
                    if num_years > 1 and self.isTimeSeriesDisplay:
                        first_year = ds.time.dt.year[0]
                        for yr, ds1 in ds.groupby('time.year'):
                            if ds1.time.dt.year[0] % 4 == 0:
                                days_in_year = 366
                            else:
                                days_in_year = 365
                            year_diff = ds1.time.dt.year - first_year
                            time_diff = np.array(
                                [np.timedelta64(x * days_in_year, 'D') for x in year_diff.values]
                            )
                            ds1['time'] = ds1.time - time_diff
                            self.display._ds[key + '%d_%d' % (k, yr)] = ds1
                            func(dsname=key + '%d_%d' % (k, yr), label=str(yr), **kwargs)
                            self.mapping[key + '%d_%d' % (k, yr)] = subplot_index
                            self.xlims[key + '%d_%d' % (k, yr)] = (
                                ds1.time.values.min(),
                                ds1.time.values.max(),
                            )
                        del self.display._ds[key + '_%d' % k]
                    else:
                        func(dsname=key + '_%d' % k, **kwargs)
                        self.mapping[key + '_%d' % k] = subplot_index
                        if self.isTimeSeriesDisplay:
                            self.xlims[key + '_%d' % k] = (
                                ds.time.values.min(),
                                ds.time.values.max(),
                            )
                    i = i + 1

        if wrap_around is False and i < np.prod(subplot_shape):
            while i < np.prod(subplot_shape):
                if len(subplot_shape) == 2:
                    subplot_index = (int(i / subplot_shape[1]), i % subplot_shape[1])
                else:
                    subplot_index = (i % subplot_shape[0],)
                self.display.axes[subplot_index].axis('off')
                i = i + 1

        for i in range(1, np.prod(subplot_shape)):
            if len(subplot_shape) == 2:
                subplot_index = (int(i / subplot_shape[1]), i % subplot_shape[1])
            else:
                subplot_index = (i % subplot_shape[0],)

            try:
                self.display.axes[subplot_index].get_legend().remove()
            except AttributeError:
                pass
        if self.isTimeSeriesDisplay:
            key_list = list(self.display._ds.keys())
            for k in key_list:
                time_min, time_max = self.xlims[k]
                subplot_index = self.mapping[k]
                self.display.set_xrng([time_min, time_max], subplot_index)

        self.display._ds = old_ds

        return self.display.axes
