"""
act.plotting
============

Class for creating timeseries plots from ACT datasets.

"""
# Import third party libraries
import matplotlib.pyplot as plt
import astral
import numpy as np
import warnings
import xarray as xr
import pandas as pd

try:
    import metpy.calc as mpcalc
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False

# Import Local Libs
from . import common
from ..utils import datetime_utils as dt_utils
from ..utils import data_utils
from copy import deepcopy

# from datetime import datetime
from scipy.interpolate import NearestNDInterpolator
if METPY_AVAILABLE:
    from metpy.units import units
    from metpy.plots import SkewT

class Display(object):
    """
    This class is the base class for all of the other Display object
    types in ACT. This contains the common attributes and routines
    between the differing *Display* classes. We recommend that you
    use the classes inherited from Display for making your plots
    such as :func:`act.plotting.TimeSeriesDisplay` and
    :func:'act.plotting.WindRoseDisplay` instead of
    trying to do so using the Display object.

    However, we do ask that if you add another object to the plotting
    module of ACT that you make it a subclass of Display. Display provides
    some basic functionality for the handling of datasets and subplot
    parameters.

    Attributes
    ----------
    fields: dict
        The dictionary containing the fields inside the ARM dataset. Each field
        has a key that links to an xarray DataArray object.
    ds: str
        The name of the datastream.
    file_dates: list
        The dates of each file being displayed.
    fig: matplotlib figure handle
        The matplotlib figure handle to display the plots on. Initializing the
        class with this set to None will create a new figure handle. See the
        matplotlib documentation on what keyword arguments are
        available.
    axes: list
        The list of axes handles to each subplot.
    plot_vars: list
        The list of variables being plotted.
    cbs: list
        The list of colorbar handles.

    Parameters
    ----------
    arm_obj: ACT Dataset, dict, or tuple
        The ACT Dataset to display in the object. If more than one dataset
        is to be specified, then a tuple can be used if all of the datasets
        conform to ARM standards. Otherwise, a dict with a key corresponding
        to the name of each datastream will need to be supplied in order
        to create the ability to plot multiple datasets.
    subplot_shape: 1 or 2D tuple
        A tuple representing the number of (rows, columns) for the subplots
        in the display. If this is None, the figure and axes will not
        be initialized.
    ds_name: str or None
        The name of the datastream to plot. This is only used if a non-ARM
        compliant dataset is being loaded and if only one such dataset is
        loaded.
    subplot_kw: dict, optional
        The kwargs to pass into :func:`fig.subplots`
    **kwargs:
        Keyword arguments passed to :func:`plt.figure`.

    """
    def __init__(self, arm_obj, subplot_shape=(1,), ds_name=None,
                 subplot_kw=None, **kwargs):
        if isinstance(arm_obj, xr.Dataset):
            if arm_obj.act.datastream is not None:
                self._arm = {arm_obj.act.datastream: arm_obj}
            elif ds_name is not None:
                self._arm = {ds_name: arm_obj}
            else:
                warnings.warn(("Could not discern datastream" +
                               "name and dict or tuple were " +
                               "not provided. Using default" +
                               "name of act_datastream!"), UserWarning)

                self._arm = {'act_datastream': arm_obj}

        # Automatically name by datastream if a tuple of object is supplied
        if isinstance(arm_obj, tuple):
            self._arm = {}
            for arm_objs in arm_obj:
                self._arm[arm_objs.act.datastream] = arm_objs

        if isinstance(arm_obj, dict):
            self._arm = arm_obj

        self.fields = {}
        self.ds = {}
        self.file_dates = {}
        self.xrng = np.zeros((1, 2))
        self.yrng = np.zeros((1, 2))

        for dsname in self._arm.keys():
            self.fields[dsname] = self._arm[dsname].variables
            if self._arm[dsname].act.datastream is not None:
                self.ds[dsname] = str(self._arm[dsname].act.datastream)
            else:
                self.ds[dsname] = "act_datastream"
            self.file_dates[dsname] = self._arm[dsname].act.file_dates

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
        subplot_shape: 1 or 2D tuple, list, or array
            The structure of the subplots in (rows, cols).
        subplot_kw: dict, optional
            The kwargs to pass into fig.subplots.
        **kwargs: keyword arguments
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
        Display: Display object or subclass
            The Display object to add as a subplot
        subplot_index: tuple
            Which subplot to add the Display to.

        Returns
        -------
        ax: matplotlib axis handle
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
            (second_value - 1)*the_shape[0] + subplot_index[0] + 1,
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
        fig: matplotlib figure handle
            The figure to place the time series display in
        ax: axis handle
            The axis handle to place the plot in

        """
        if self.fig is not None:
            plt.close(self.fig)
            del self.fig

        del self.axes
        self.fig = fig
        self.axes = np.array([ax])

    def add_colorbar(self, mappable, title=None, subplot_index=(0, )):
        """
        Adds a colorbar to the plot

        Parameters
        ----------
        mappable: matplotlib mappable
            The mappable to base the colorbar on.
        title: str
            The title of the colorbar. Set to None to have no title.
        subplot_index: 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.

        Returns
        -------
        cbar: matplotlib colorbar handle
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


class TimeSeriesDisplay(Display):
    """
    This subclass contains routines that are specific to plotting
    time series plots from data. It is inherited from Display and therefore
    contains all of Display's attributes and methods.

    Examples
    --------

    To create a TimeSeriesDisplay with 3 rows, simply do:

    .. code-block:: python

        ds = act.read_netcdf(the_file)
        disp = act.plotting.TimeSeriesDisplay(
           ds, subplot_shape=(3,), figsize=(15,5))

    The TimeSeriesDisplay constructor takes in the same keyword arguments as
    plt.subplots. For more information on the plt.subplots keyword arguments,
    see the `matplotlib documentation
    <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html>`_.
    If no subplot_shape is provided, then no figure or axis will be created
    until add_subplots or plots is called.

    """
    def __init__(self, arm_obj, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(arm_obj, subplot_shape, ds_name, **kwargs)

    def day_night_background(self, dsname=None, subplot_index=(0, )):
        """
        Colorcodes the background according to sunrise/sunset

        Parameters
        ----------
        dsname: None or str
            If there is more than one datastream in the display object the
            name of the datastream needs to be specified. If set to None and
            there is only one datastream then ACT will use the sole datastream
            in the object.
        subplot_index: 1 or 2D tuple, list, or array
            The index to the subplot to place the day and night background in.

        """

        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream to derive the " +
                              "information needed for the day and night " +
                              "background when 2 or more datasets are in " +
                              "the display object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Get File Dates
        file_dates = self._arm[dsname].act.file_dates
        if len(file_dates) == 0:
            sdate = dt_utils.numpy_to_arm_date(
                self._arm[dsname].time.values[0])
            edate = dt_utils.numpy_to_arm_date(
                self._arm[dsname].time.values[-1])
            file_dates = [sdate, edate]

        all_dates = dt_utils.dates_between(file_dates[0], file_dates[-1])

        if self.axes is None:
            raise RuntimeError("day_night_background requires the plot to "
                               "be displayed.")

        ax = self.axes[subplot_index]

        # initialize the plot to a gray background for total darkness
        rect = ax.patch
        rect.set_facecolor('0.85')

        # Initiate Astral Instance
        a = astral.Astral()
        if self._arm[dsname].lat.data.size > 1:
            lat = self._arm[dsname].lat.data[0]
            lon = self._arm[dsname].lon.data[0]
        else:
            lat = float(self._arm[dsname].lat.data)
            lon = float(self._arm[dsname].lon.data)

        for f in all_dates:
            sun = a.sun_utc(f, lat, lon)
            # add yellow background for specified time period
            ax.axvspan(sun['sunrise'], sun['sunset'], facecolor='#FFFFCC')

            # add local solar noon line
            ax.axvline(x=sun['noon'], linestyle='--', color='y')

    def set_xrng(self, xrng, subplot_index=(0, )):
        """
        Sets the x range of the plot.

        Parameters
        ----------
        xrng: 2 number array
            The x limits of the plot.
        subplot_index: 1 or 2D tuple, list, or array
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
        self.xrng[subplot_index, :] = np.array(xrng, dtype='datetime64[D]')

    def set_yrng(self, yrng, subplot_index=(0, )):
        """
        Sets the y range of the plot.

        Parameters
        ----------
        yrng: 2 number array
            The y limits of the plot.
        subplot_index: 1 or 2D tuple, list, or array
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

    def plot(self, field, dsname=None, subplot_index=(0, ),
             cmap=None, cbmin=None, cbmax=None, set_title=None,
             add_nan=False, day_night_background=False,
             invert_y_axis=False,
             **kwargs):
        """
        Makes a timeseries plot. If subplots have not been added yet, an axis
        will be created assuming that there is only going to be one plot.

        Parameters
        ----------
        field: str
            The name of the field to plot
        dsname: None or str
            If there is more than one datastream in the display object the
            name of the datastream needs to be specified. If set to None and
            there is only one datastream ACT will use the sole datastream
            in the object.
        subplot_index: 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        cmap: matplotlib colormap
            The colormap to use.
        cbmin: float
            The minimum for the colorbar. This is not used for 1D plots.
        cbmax: float
            The maximum for the colorbar. This is not used for 1D plots.
        set_title: str
            The title for the plot.
        add_nan: bool
            Set to True to fill in data gaps with NaNs.
        day_night_background: bool
            Set to True to fill in a color coded background
            according to the time of day.
        kwargs: dict
            The keyword arguments for :func:`plt.plot` (1D timeseries) or
            :func:`plt.pcolormesh` (2D timeseries).

        Returns
        -------
        ax: matplotlib axis handle
            The matplotlib axis handle of the plot.
        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Get data and dimensions
        data = self._arm[dsname][field]
        dim = list(self._arm[dsname][field].dims)
        xdata = self._arm[dsname][dim[0]]

        if 'units' in data.attrs:
            ytitle = ''.join(['(', data.attrs['units'], ')'])
        else:
            ytitle = field

        if len(dim) > 1:
            ydata = self._arm[dsname][dim[1]]
            units = ytitle
            ytitle = ''.join(['(', ydata.attrs['units'], ')'])
        else:
            ydata = None

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        ax = self.axes[subplot_index]

        if ydata is None:
            if day_night_background is True:
                self.day_night_background(subplot_index=subplot_index,
                                          dsname=dsname)
            self.axes[subplot_index].plot(xdata, data, '.', **kwargs)
        else:
            # Add in nans to ensure the data are not streaking
            if add_nan is True:
                xdata, data = data_utils.add_in_nan(xdata, data)
            mesh = self.axes[subplot_index].pcolormesh(
                xdata, ydata, data.transpose(),
                cmap=cmap, edgecolors='face', **kwargs)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, field, 'on',
                                 dt_utils.numpy_to_arm_date(
                                     self._arm[dsname].time.values[0])])

        self.axes[subplot_index].set_title(set_title)

        # Set YTitle
        self.axes[subplot_index].set_ylabel(ytitle)

        # Set X Limit - We want the same time axes for all subplots
        if not hasattr(self, 'time_rng'):
            self.time_rng = [xdata.min().values, xdata.max().values]

        self.set_xrng(self.time_rng, subplot_index)

        # Set Y Limit
        if hasattr(self, 'yrng'):
            # Make sure that the yrng is not just the default
            if not np.all(self.yrng[subplot_index] == 0):
                self.set_yrng(self.yrng[subplot_index], subplot_index)
            else:
                if ydata is None:
                    our_data = data.values
                else:
                    our_data = ydata
                if np.isfinite(our_data).any():
                    if invert_y_axis is False:
                        yrng = [np.nanmin(our_data), np.nanmax(our_data)]
                    else:
                        yrng = [np.nanmax(our_data), np.nanmin(our_data)]
                else:
                    yrng = [0, 1]
                self.set_yrng(yrng, subplot_index)

        # Set X Format
        if len(subplot_index) == 1:
            days = (self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0])
        else:
            days = (self.xrng[subplot_index[0], subplot_index[1], 1] -
                    self.xrng[subplot_index[0], subplot_index[1], 0])

        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            self.axes[subplot_index].set_xlabel('Time [UTC]')

        if ydata is not None:
            self.add_colorbar(mesh, title=units, subplot_index=subplot_index)

        return self.axes[subplot_index]

    def plot_barbs_from_spd_dir(self, dir_field, spd_field, pres_field=None,
                                dsname=None, **kwargs):
        """
        This procedure will make a wind barb plot timeseries.
        If a pressure field is given and the wind fields are 1D, which, for
        example, would occur if one wants to plot a timeseries of
        rawinsonde data, then a time-height cross section of
        winds will be made.

        Note: this procedure calls plot_barbs_from_u_v and will take in the
        same keyword arguments as that procedure.

        Parameters
        ----------
        dir_field: str
            The name of the field specifying the wind direction in degrees.
            0 degrees is defined to be north and increases clockwise like
            what is used in standard meteorological notation.
        spd_field: str
            The name of the field specifying the wind speed in m/s.
        pres_field: str
            The name of the field specifying pressure or height. If using
            height coordinates, then we recommend setting invert_y_axis
            to False.
        dsname: str
            The name of the datastream to plot. Setting to None will make
            ACT attempt to autodetect this.
        kwargs: dict
            Any additional keyword arguments will be passed into
            :func:`act.plotting.TimeSeriesDisplay.plot_barbs_from_u_and_v`.

        Returns
        -------
        the_ax: matplotlib axis handle
            The handle to the axis where the plot was made on.

        Examples
        --------
        ..code-block :: python

            sonde_ds = act.io.armfiles.read_netcdf(
                act.tests.sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
            BarbDisplay = act.plotting.TimeSeriesDisplay(
                {'sonde_darwin': sonde_ds}, figsize=(10,5))
            BarbDisplay.plot_barbs_from_spd_dir('deg', 'wspd', 'pres',
                                                num_barbs_x=20)
        """

        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Make temporary field called tempu, tempv
        spd = self._arm[dsname][spd_field]
        dir = self._arm[dsname][dir_field]
        tempu = -np.sin(np.deg2rad(dir)) * spd
        tempv = -np.cos(np.deg2rad(dir)) * spd
        self._arm[dsname]["temp_u"] = deepcopy(self._arm[dsname][spd_field])
        self._arm[dsname]["temp_v"] = deepcopy(self._arm[dsname][spd_field])
        self._arm[dsname]["temp_u"].values = tempu
        self._arm[dsname]["temp_v"].values = tempv
        the_ax = self.plot_barbs_from_u_v("temp_u", "temp_v", pres_field,
                                          dsname, **kwargs)
        del self._arm[dsname]["temp_u"], self._arm[dsname]["temp_v"]
        return the_ax

    def plot_barbs_from_u_v(self, u_field, v_field, pres_field=None,
                            dsname=None, subplot_index=(0, ),
                            set_title=None,
                            day_night_background=False,
                            invert_y_axis=True,
                            num_barbs_x=20, num_barbs_y=20, **kwargs):
        """
        This function will plot a wind barb timeseries from u and v wind
        data. If pres_field is given, a time-height series will be plotted
        from 1-D wind data.

        Parameters
        ----------
        u_field: str
            The name of the field containing the U component of the wind.
        v_field: str
            The name of the field containing the V component of the wind.
        pres_field: str or None
            The name of the field containing the pressure or height. Set
            to None to not use this.
        dsname: str or None
            The name of the datastream to plot. Setting to None will make
            ACT automatically try to determine this.
        subplot_index: 2-tuple
            The index of the subplot to make the plot on.
        set_title: str or None
            The title of the plot.
        day_night_background: bool
            Set to True to plot a day/night background.
        invert_y_axis: bool
            Set to True to invert the y axis (i.e. for plotting pressure as
            the height coordinate).
        num_barbs_x: int
            The number of wind barbs to plot in the x axis.
        num_barbs_y: int
            The number of wind barbs to plot in the y axis.
        kwargs: dict
            Additional keyword arguments will be passed into plt.barbs.

        Returns
        -------
        ax: matplotlib axis handle
             The axis handle that contains the reference to the
             constructed plot.
        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Get data and dimensions
        u = self._arm[dsname][u_field].values
        v = self._arm[dsname][v_field].values
        dim = list(self._arm[dsname][u_field].dims)
        xdata = self._arm[dsname][dim[0]].values
        num_x = xdata.shape[-1]
        barb_step_x = round(num_x / num_barbs_x)
        if len(dim) > 1 and pres_field is None:
            ydata = self._arm[dsname][dim[1]]
            ytitle = ''.join(['(', ydata.attrs['units'], ')'])
            units = ytitle
            num_y = ydata.shape[0]
            barb_step_y = round(num_y / num_barbs_y)
        elif pres_field is not None:
            # What we will do here is do a nearest-neighbor interpolation
            # for each member of the series. Coordinates are time, pressure
            pres = self._arm[dsname][pres_field]
            u_interp = NearestNDInterpolator(
                (xdata, pres.values), u, rescale=True)
            v_interp = NearestNDInterpolator(
                (xdata, pres.values), v, rescale=True)
            barb_step_x = 1
            barb_step_y = 1
            x_times = pd.date_range(xdata.min(), xdata.max(),
                                    periods=num_barbs_x)
            if num_barbs_y == 1:
                y_levels = pres.mean()
            else:
                y_levels = np.linspace(np.nanmin(pres), np.nanmax(pres),
                                       num_barbs_y)
            xdata, ydata = np.meshgrid(x_times, y_levels, indexing='ij')
            u = u_interp(xdata, ydata)
            v = v_interp(xdata, ydata)
            ytitle = ''.join(['(', pres.attrs['units'], ')'])
        else:
            ydata = None

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        if ydata is None:
            ydata = np.ones(xdata.shape)
            self.axes[subplot_index].barbs(xdata[::barb_step_x],
                                           ydata[::barb_step_x],
                                           u[::barb_step_x],
                                           v[::barb_step_x],
                                           **kwargs)
            self.axes[subplot_index].set_yticks([])
        else:
            self.axes[subplot_index].barbs(
                xdata[::barb_step_y, ::barb_step_x],
                ydata[::barb_step_y, ::barb_step_x],
                u[::barb_step_y, ::barb_step_x],
                v[::barb_step_y, ::barb_step_x],
                **kwargs)

        if day_night_background is True:
            self.day_night_background(subplot_index)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, 'on',
                                 dt_utils.numpy_to_arm_date(
                                     self._arm[dsname].time.values[0])])

        self.axes[subplot_index].set_title(set_title)

        # Set YTitle
        if 'ytitle' in locals():
            self.axes[subplot_index].set_ylabel(ytitle)

        # Set X Limit - We want the same time axes for all subplots
        time_rng = [xdata.min(), xdata.max()]
        self.set_xrng(time_rng, subplot_index)

        # Set Y Limit
        if hasattr(self, 'yrng'):
            # Make sure that the yrng is not just the default
            if not np.all(self.yrng[subplot_index] == 0):
                self.set_yrng(self.yrng[subplot_index], subplot_index)
            else:
                if ydata is None:
                    our_data = xdata
                else:
                    our_data = ydata
                if np.isfinite(our_data).any():
                    if invert_y_axis is False:
                        yrng = [np.nanmin(our_data), np.nanmax(our_data)]
                    else:
                        yrng = [np.nanmax(our_data), np.nanmin(our_data)]
                else:
                    yrng = [0, 1]
                self.set_yrng(yrng, subplot_index)

        # Set X Format
        if len(subplot_index) == 1:
            days = (self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0])
        else:
            days = (self.xrng[subplot_index[0], subplot_index[1], 1] -
                    self.xrng[subplot_index[0], subplot_index[1], 0])

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            self.axes[subplot_index].set_xlabel('Time [UTC]')

        myFmt = common.get_date_format(days)
        self.axes[subplot_index].xaxis.set_major_formatter(myFmt)
        return self.axes[subplot_index]

    def plot_time_height_xsection_from_1d_data(
            self, data_field, pres_field, dsname=None, subplot_index=(0, ),
            set_title=None, day_night_background=False, num_time_periods=20,
            num_y_levels=20, cbmin=None, cbmax=None, invert_y_axis=True,
            **kwargs):
        """
        This will plot a time-height cross section from 1D datasets using
        nearest neighbor interpolation on a regular time by height grid.
        All that is needed are a data variable and a height variable.

        Parameters
        ----------
        data_field: str
            The name of the field to plot.
        pres_field: str
            The name of the height or pressure field to plot.
        dsname: str or None
            The name of the datastream to plot
        subplot_index: 2-tuple
            The index of the subplot to create the plot on.
        set_title: str or None
            The title of the plot.
        day_night_background: bool
            Set to true to plot the day/night background
        num_time_periods: int
            Set to determine how many time periods. Setting to None
            will do one time period per day.
        num_y_levels: int
            The number of levels in the y axis to use
        cbmin: float
            The minimum for the colorbar.
        cbmax: float
            The maximum for the colorbar.
        invert_y_axis: bool
             Set to true to invert the y-axis (recommended for
             pressure coordinates)
        kwargs: dict
             Additional keyword arguments will be passed
             into :func:`plt.pcolormesh`

        Returns
        -------
        ax: matplotlib axis handle
            The matplotlib axis handle pointing to the plot.
        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2"
                              "or more datasets in the TimeSeriesDisplay"
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        dim = list(self._arm[dsname][data_field].dims)
        if len(dim) > 1:
            raise ValueError(("plot_time_height_xsection_from_1d_data only "
                              "supports 1-D datasets. For datasets with 2 or "
                              "more dimensions use plot()."))

        # Get data and dimensions
        data = self._arm[dsname][data_field].values
        xdata = self._arm[dsname][dim[0]].values

        # What we will do here is do a nearest-neighbor interpolation for each
        # member of the series. Coordinates are time, pressure
        pres = self._arm[dsname][pres_field]
        u_interp = NearestNDInterpolator(
            (xdata, pres.values), data, rescale=True)
        # Mask points where we have no data
        # Count number of unique days
        x_times = pd.date_range(xdata.min(), xdata.max(),
                                periods=num_time_periods)
        y_levels = np.linspace(np.nanmin(pres), np.nanmax(pres), num_y_levels)
        tdata, ydata = np.meshgrid(x_times, y_levels, indexing='ij')
        data = u_interp(tdata, ydata)
        ytitle = ''.join(['(', pres.attrs['units'], ')'])
        units = (data_field + ' (' +
                 self._arm[dsname][data_field].attrs['units'] + ')')

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        mesh = self.axes[subplot_index].pcolormesh(
            tdata, ydata, data, **kwargs)

        if day_night_background is True:
            self.day_night_background(subplot_index)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [dsname, 'on',
                 dt_utils.numpy_to_arm_date(self._arm[dsname].time.values[0])])

        self.axes[subplot_index].set_title(set_title)

        # Set YTitle
        if 'ytitle' in locals():
            self.axes[subplot_index].set_ylabel(ytitle)

        # Set X Limit - We want the same time axes for all subplots
        time_rng = [x_times[-1], x_times[0]]

        self.set_xrng(time_rng, subplot_index)

        # Set Y Limit
        if hasattr(self, 'yrng'):
            # Make sure that the yrng is not just the default
            if not np.all(self.yrng[subplot_index] == 0):
                self.set_yrng(self.yrng[subplot_index], subplot_index)
            else:
                if ydata is None:
                    our_data = data.values
                else:
                    our_data = ydata
                if np.isfinite(our_data).any():
                    if invert_y_axis is False:
                        yrng = [np.nanmin(our_data), np.nanmax(our_data)]
                    else:
                        yrng = [np.nanmax(our_data), np.nanmin(our_data)]
                else:
                    yrng = [0, 1]
                self.set_yrng(yrng, subplot_index)

        # Set X Format
        if len(subplot_index) == 1:
            days = (self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0])
        else:
            days = (self.xrng[subplot_index[0], subplot_index[1], 1] -
                    self.xrng[subplot_index[0], subplot_index[1], 0])

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            self.axes[subplot_index].set_xlabel('Time [UTC]')

        if ydata is not None:
            self.add_colorbar(mesh, title=units, subplot_index=subplot_index)

        myFmt = common.get_date_format(days)
        self.axes[subplot_index].xaxis.set_major_formatter(myFmt)

        return self.axes[subplot_index]


class WindRoseDisplay(Display):
    """
    A class for handing wind rose plots.

    This is inherited from the :func:`act.plotting.Display`
    class and has therefore has the same attributes as that class.
    See :func:`act.plotting.Display`
    for more information.  There are no additional attributes or parameters
    to this class.

    Examples
    --------

    To create a WindRoseDisplay object, simply do:

    .. code-block :: python

        sonde_ds = act.io.armfiles.read_netcdf('sonde_data.nc')
        WindDisplay = act.plotting.WindRoseDisplay(sonde_ds, figsize=(8,10))

    """
    def __init__(self, arm_obj, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(arm_obj, subplot_shape, ds_name,
                         subplot_kw=dict(projection='polar'), **kwargs)

    def set_thetarng(self, trng=(0., 360.), subplot_index=(0,)):
        """
        Sets the theta range of the wind rose plot.

        Parameters
        ----------
        trng: 2-tuple
            The range (in degrees).
        subplot_index: 2-tuple
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
        rrng: 2-tuple
            The range for the plot radius (in %).
        subplot_index: 2-tuple
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
        dir_field: str
            The name of the field representing the wind direction (in degrees).
        spd_field: str
            The name of the field representing the wind speed.
        dsname: str
            The name of the datastream to plot from. Set to None to
            let ACT automatically try to determine this.
        subplot_index: 2-tuple
            The index of the subplot to place the plot on
        cmap: str or matplotlib colormap
            The name of the matplotlib colormap to use.
        set_title: str
            The title of the plot.
        num_dirs: int
            The number of directions to split the wind rose into.
        spd_bins: 1D array-like
            The bin boundaries to sort the wind speeds into.
        tick_interval:
            The interval (in %) for the ticks on the radial axis.
        kwargs:
            Additional keyword arguments will be passed into :func:plt.bar

        Returns
        -------
        ax: matplotlib axis handle
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
                the_range = np.logical_or(dir_data < deg_width / 2.,
                                          dir_data > 360. - deg_width / 2.)
            else:
                the_range = np.logical_and(
                    dir_data >= dir_bins_mid[i] - deg_width / 2,
                    dir_data <= dir_bins_mid[i] + deg_width / 2)
            hist, bins = np.histogram(spd_data[the_range], spd_bins)
            wind_hist[i] = hist

        wind_hist = wind_hist / np.sum(wind_hist) * 100
        mins = np.deg2rad(dir_bins_mid)
        # Do the first level
        units = self._arm[dsname][spd_field].attrs['units']
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

class SkewTDisplay(Display):
    """
    A class for making Skew-T plots.

    his is inherited from the :func:`act.plotting.Display`
    class and has therefore has the same attributes as that class.
    See :func:`act.plotting.Display`
    for more information.  There are no additional attributes or parameters
    to this class.

    In order to create Skew-T plots, ACT needs the MetPy package to be
    installed on your system. More information about
    MetPy go here: https://unidata.github.io/MetPy/latest/index.html.

    Examples
    --------
    sonde_ds = act.io.armfiles.read_netcdf(
       act.tests.sample_files.EXAMPLE_SONDE1)

    skewt = act.plotting.SkewTDisplay(sonde_ds)

    skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
    plt.show()

    """
    def __init__(self, arm_obj, subplot_shape=(1,), ds_name=None, **kwargs):
        # We want to use our routine to handle subplot adding, not the main
        # one
        if not METPY_AVAILABLE:
            raise ImportError("MetPy need to be installed on your system to " +
                              "make Skew-T plots.")
        new_kwargs = kwargs.copy()
        super().__init__(arm_obj, None, ds_name,
                         subplot_kw=dict(projection='skewx'), **new_kwargs)

        # Make a SkewT object for each subplot
        self.add_subplots(subplot_shape)

    def add_subplots(self, subplot_shape=(1,), **kwargs):
        """
        Adds subplots to the Display object. The current
        figure in the object will be deleted and overwritten.

        Parameters
        ----------
        subplot_shape: 1 or 2D tuple, list, or array
            The structure of the subplots in (rows, cols).
        subplot_kw: dict, optional
            The kwargs to pass into fig.subplots.
        **kwargs: keyword arguments
            Any other keyword arguments that will be passed
            into :func:`matplotlib.pyplot.figure` when the figure
            is made. The figure is only made if the *fig*
            property is None. See the matplotlib
            documentation for further details on what keyword
            arguments are available.
        """
        del self.axes
        if self.fig is None:
            self.fig = plt.figure(**kwargs)
        self.SkewT = np.empty(shape=subplot_shape, dtype=SkewT)
        self.axes = np.empty(shape=subplot_shape, dtype=plt.Axes)
        if len(subplot_shape) == 1:
            for i in range(subplot_shape[0]):
                subplot_tuple = (subplot_shape[0], 1, i+1)
                self.SkewT[i] = SkewT(fig=self.fig, subplot=subplot_tuple)
                self.axes[i] = self.SkewT[i].ax
        elif len(subplot_shape) == 2:
            for i in range(subplot_shape[0]):
                for j in range(subplot_shape[1]):
                    subplot_tuple = (subplot_shape[0],
                                     subplot_shape[1],
                                     i*subplot_shape[1]+j+1)
                    self.SkewT[i] = SkewT(fig=self.fig, subplot=subplot_tuple)
                    self.axes[i] = self.SkewT[i].ax
        else:
            raise ValueError("Subplot shape must be 1 or 2D!")

    def set_xrng(self, xrng, subplot_index=(0,)):
        """
        Sets the x range of the plot.

        Parameters
        ----------
        xrng: 2 number array
            The x limits of the plot.
        subplot_index: 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.

        """
        if self.axes is None:
            raise RuntimeError("set_xrng requires the plot to be displayed.")

        if not hasattr(self, 'xrng') and len(self.axes.shape) == 2:
            self.xrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2))
        elif not hasattr(self, 'xrng') and len(self.axes.shape) == 1:
            self.xrng = np.zeros((self.axes.shape[0], 2))

        self.axes[subplot_index].set_xlim(xrng)
        self.xrng[subplot_index, :] = np.array(xrng)

    def set_yrng(self, yrng, subplot_index=(0,)):
        """
        Sets the y range of the plot.

        Parameters
        ----------
        yrng: 2 number array
            The y limits of the plot.
        subplot_index: 1 or 2D tuple, list, or array
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

    def plot_from_spd_and_dir(self, spd_field, dir_field,
                              p_field, t_field, td_field, dsname=None,
                              **kwargs):
        """
        This plot will make a sounding plot from wind data that is given
        in speed and direction.

        Parameters
        ----------
        spd_field: str
            The name of the field corresponding to the wind speed.
        dir_field: str
            The name of the field corresponding to the wind direction
            in degrees from North.
        p_field: str
            The name of the field containing the atmospheric pressure.
        t_field: str
            The name of the field containing the atmospheric temperature.
        td_field: str
            The name of the field containing the dewpoint
        dsname: str or None
             The name of the datastream to plot. Set to None to make ACT
            attempt to automatically determine this.

        Additional keyword arguments will be passed into
        :func:`act.plotting.SkewTDisplay.plot_from_u_and_v`

        Returns
        -------
        ax: matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.
        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Make temporary field called tempu, tempv
        spd = self._arm[dsname][spd_field]
        dir = self._arm[dsname][dir_field]
        tempu = -np.sin(np.deg2rad(dir)) * spd
        tempv = -np.cos(np.deg2rad(dir)) * spd
        self._arm[dsname]["temp_u"] = deepcopy(self._arm[dsname][spd_field])
        self._arm[dsname]["temp_v"] = deepcopy(self._arm[dsname][spd_field])
        self._arm[dsname]["temp_u"].values = tempu
        self._arm[dsname]["temp_v"].values = tempv
        the_ax = self.plot_from_u_v("temp_u", "temp_v", pres_field,
                                          t_field, td_field, dsname, **kwargs)
        del self._arm[dsname]["temp_u"], self._arm[dsname]["temp_v"]
        return the_ax

    def plot_from_u_and_v(self, u_field, v_field, p_field,
                          t_field, td_field, dsname=None, subplot_index=(0,),
                          p_levels_to_plot=None, show_parcel=True,
                          shade_cape=True, shade_cin=True, set_title=None,
                          plot_barbs_kwargs=dict(), plot_kwargs=dict(),):
        """
        This function will plot a Skew-T from a sounding dataset. The wind
        data must be given in u and v.

        Parameters
        ----------
        u_field: str
            The name of the field containing the u component of the wind.
        v_field: str
            The name of the field containing the v component of the wind.
        p_field: str
            The name of the field containing the pressure.
        t_field: str
            The name of the field containing the temperature.
        td_field: str
            The name of the field containing the dewpoint temperature.
        dsname: str or None
            The name of the datastream to plot. Set to None to make ACT
            attempt to automatically determine this.
        subplot_index: tuple
            The index of the subplot to make the plot on.
        p_levels_to_plot: 1D array
            The pressure levels to plot the wind barbs on. Set to None
            to have ACT to use neatly spaced defaults of
            50, 100, 200, 300, 400, 500, 600, 700, 750, 800,
            850, 900, 950, and 1000 hPa.
        show_parcel: bool
            Set to True to show the temperature of a parcel lifted
            from the surface.
        shade_cape: bool
            Set to True to shade the CAPE red.
        shade_cin: bool
            Set to True to shade the CIN blue.
        set_title: None or str
            The title of the plot is set to this. Set to None to use
            a default title.
        plot_barbs_kwargs: dict
            Additional keyword arguments to pass into MetPy's
            SkewT.plot_barbs.
        plot_kwargs: dict
            Additional keyword arguments to pass into MetPy's
            SkewT.plot.

        Returns
        -------
        ax: matplotlib axis handle
            The axis handle to the plot.
        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        if p_levels_to_plot == None:
            p_levels_to_plot = np.array([50., 100., 200., 300., 400.,
                                         500., 600., 700., 750., 800.,
                                         850., 900., 950., 1000.])
        T = self._arm[dsname][t_field]
        T_units = self._arm[dsname][t_field].attrs["units"]
        if T_units == "C":
            T_units = "degC"

        T = T.values * getattr(units, T_units)
        Td = self._arm[dsname][td_field]
        Td_units = self._arm[dsname][td_field].attrs["units"]
        if Td_units == "C":
            Td_units = "degC"

        Td = Td.values * getattr(units, Td_units)
        u = self._arm[dsname][u_field]
        u_units = self._arm[dsname][u_field].attrs["units"]
        u = u.values * getattr(units, u_units)

        v = self._arm[dsname][v_field]
        v_units = self._arm[dsname][v_field].attrs["units"]
        v = v.values * getattr(units, v_units)

        p = self._arm[dsname][p_field]
        p_units = self._arm[dsname][p_field].attrs["units"]
        p = p.values * getattr(units, p_units)

        u_red = np.zeros_like(p_levels_to_plot) * getattr(units, u_units)
        v_red = np.zeros_like(p_levels_to_plot) * getattr(units, v_units)

        for i in range(len(p_levels_to_plot)):
            index = np.argmin(np.abs(p_levels_to_plot[i] - p))
            u_red[i] = u[index].magnitude * getattr(units, u_units)
            v_red[i] = v[index].magnitude * getattr(units, v_units)

        p_levels_to_plot = p_levels_to_plot * getattr(units, p_units)
        self.SkewT[subplot_index].plot(p, T, 'r', **plot_kwargs)
        self.SkewT[subplot_index].plot(p, Td, 'g', **plot_kwargs)
        self.SkewT[subplot_index].plot_barbs(
            p_levels_to_plot, u_red, v_red, **plot_barbs_kwargs)

        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        where_unstable = np.greater(prof, -60 * units.degC)
        if show_parcel:
            # Only plot where prof > T
            lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
            self.SkewT[subplot_index].plot(
                lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black',
                **plot_kwargs)
            self.SkewT[subplot_index].plot(
                p, prof, 'k', linewidth=2, **plot_kwargs)

        if shade_cape:
            self.SkewT[subplot_index].shade_cape(
                p, T, prof, linewidth=2)

        if shade_cin:
            self.SkewT[subplot_index].shade_cin(
                p, T, prof, linewidth=2)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [dsname, 'on',
                dt_utils.numpy_to_arm_date(self._arm[dsname].time.values[0])])

        self.axes[subplot_index].set_title(set_title)

        # Set Y Limit
        if hasattr(self, 'yrng'):
            # Make sure that the yrng is not just the default
            if not np.all(self.yrng[subplot_index] == 0):
                self.set_yrng(self.yrng[subplot_index], subplot_index)
            else:
                our_data = p.magnitude
                if np.isfinite(our_data).any():
                    yrng = [np.nanmax(our_data), np.nanmin(our_data)]
                else:
                    yrng = [1000., 100.]
                self.set_yrng(yrng, subplot_index)

        # Set X Limit
        xrng = [T.magnitude.min()-10., T.magnitude.max()+10.]
        self.set_xrng(xrng, subplot_index)

        return self.axes[subplot_index]
