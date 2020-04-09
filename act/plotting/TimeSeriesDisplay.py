"""
act.plotting.TimeSeriesDisplay
------------------------------

Stores the class for TimeSeriesDisplay.

"""

import matplotlib.pyplot as plt
import astral
import numpy as np
import pandas as pd
import datetime as dt
import warnings
from re import search as re_search

from .plot import Display
# Import Local Libs
from . import common
from ..utils import datetime_utils as dt_utils
from ..utils.datetime_utils import reduce_time_ranges, determine_time_delta
from ..qc.qcfilter import parse_bit
from ..utils import data_utils
from copy import deepcopy
from scipy.interpolate import NearestNDInterpolator


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
    def __init__(self, obj, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(obj, subplot_shape, ds_name, **kwargs)

    def day_night_background(self, dsname=None, subplot_index=(0, )):
        """
        Colorcodes the background according to sunrise/sunset.

        Parameters
        ----------
        dsname : None or str
            If there is more than one datastream in the display object the
            name of the datastream needs to be specified. If set to None and
            there is only one datastream then ACT will use the sole datastream
            in the object.
        subplot_index : 1 or 2D tuple, list, or array
            The index to the subplot to place the day and night background in.

        Returns
        -------
        None

        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream to derive the " +
                              "information needed for the day and night " +
                              "background when 2 or more datasets are in " +
                              "the display object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Get File Dates
        try:
            file_dates = self._arm[dsname].attrs['_file_dates']
        except KeyError:
            file_dates = []
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

        # Find variable names for latitude and longitude
        variables = list(self._arm[dsname].data_vars)
        lat_name = [var for var in ['lat', 'latitude'] if var in variables]
        lon_name = [var for var in ['lon', 'longitude'] if var in variables]
        if len(lat_name) == 0:
            lat_name = None
        else:
            lat_name = lat_name[0]

        if len(lon_name) == 0:
            lon_name = None
        else:
            lon_name = lon_name[0]

        # Variable name does not match, look for standard_name declaration
        if lat_name is None or lon_name is None:
            for var in variables:
                try:
                    if self._arm[dsname][var].attrs['standard_name'] == 'latitude':
                        lat_name = var
                except KeyError:
                    pass

                try:
                    if self._arm[dsname][var].attrs['standard_name'] == 'longitude':
                        lon_name = var
                except KeyError:
                    pass

                if lat_name is not None and lon_name is not None:
                    break

        if lat_name is None or lon_name is None:
            return

        try:
            if self._arm[dsname].lat.data.size > 1:
                # Look for non-NaN values to use for locaiton. If not found use first value.
                lat = self._arm[dsname][lat_name].values
                index = np.where(np.isfinite(lat))[0]
                if index.size == 0:
                    index = [0]
                lat = float(lat[index[0]])
                # Look for non-NaN values to use for locaiton. If not found use first value.
                lon = self._arm[dsname][lon_name].values
                index = np.where(np.isfinite(lon))[0]
                if index.size == 0:
                    index = [0]
                lon = float(lon[index[0]])
            else:
                lat = float(self._arm[dsname][lat_name].values)
                lon = float(self._arm[dsname][lon_name].values)
        except AttributeError:
            return

        if not np.isfinite(lat):
            warnings.warn(f"Latitude value in dataset of '{lat}' is not finite. ",
                          RuntimeWarning)
            return

        if not np.isfinite(lon):
            warnings.warn(f"Longitude value in dataset of '{lon}' is not finite. ",
                          RuntimeWarning)
            return

        lat_range = [-90, 90]
        if not (lat_range[0] <= lat <= lat_range[1]):
            warnings.warn(f"Latitude value in dataset of '{lat}' not within acceptable "
                          f"range of {lat_range[0]} <= latitude <= {lat_range[1]}. ",
                          RuntimeWarning)
            return

        lon_range = [-180, 180]
        if not (lon_range[0] <= lon <= lon_range[1]):
            warnings.warn(f"Longitude value in dataset of '{lon}' not within acceptable "
                          f"range of {lon_range[0]} <= longitude <= {lon_range[1]}. ",
                          RuntimeWarning)
            return

        # initialize the plot to a gray background for total darkness
        rect = ax.patch
        rect.set_facecolor('0.85')

        # Initiate Astral Instance
        a = astral.Astral()
        # Set the the number of degrees the sun must be below the horizon
        # for the dawn/dusk calculation. Need to do this so when the calculation
        # sends an error it is not going to be an inacurate switch to setting
        # the full day.
        a.solar_depression = 0

        for f in all_dates:
            # Loop over previous, current and following days to cover all overlaps
            # due to local vs UTC times.
            for ii in [-1, 0, 1]:
                try:
                    new_time = f + dt.timedelta(days=ii)
                    sun = a.sun_utc(new_time, lat, lon)

                    # add yellow background for specified time period
                    ax.axvspan(
                        sun['sunrise'], sun['sunset'], facecolor='#FFFFCC', zorder=0)

                    # add local solar noon line
                    ax.axvline(x=sun['noon'], linestyle='--', color='y', zorder=1)

                except astral.AstralError:
                    # Error for all day and all night is the same. Check to see
                    # if sun is above horizon at solar noon. If so plot.
                    if a.solar_elevation(new_time, lat, lon) > 0:
                        # Make whole background yellow for when sun does not reach
                        # horizon. Use in high latitude locations.
                        ax.axvspan(dt.datetime(f.year, f.month, f.day, hour=0,
                                               minute=0, second=0),
                                   dt.datetime(f.year, f.month, f.day, hour=23,
                                               minute=59, second=59),
                                   facecolor='#FFFFCC')

                        # add local solar noon line
                        ax.axvline(
                            x=a.solar_noon_utc(f, lon), linestyle='--', color='y')

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
            raise RuntimeError("set_xrng requires the plot to be displayed.")

        if not hasattr(self, 'xrng') and len(self.axes.shape) == 2:
            self.xrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2),
                                 dtype='datetime64[D]')
        elif not hasattr(self, 'xrng') and len(self.axes.shape) == 1:
            self.xrng = np.zeros((self.axes.shape[0], 2),
                                 dtype='datetime64[D]')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.axes[subplot_index].set_xlim(xrng)
            self.xrng[subplot_index, :] = np.array(xrng, dtype='datetime64[D]')

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
             cmap=None, set_title=None,
             add_nan=False, day_night_background=False,
             invert_y_axis=False, abs_limits=(None, None), time_rng=None,
             use_var_for_y=None,
             assessment_overplot=False,
             assessment_overplot_category={'Incorrect': ['Bad', 'Incorrect'],
                                           'Suspect': ['Indeterminate', 'Suspect']},
             assessment_overplot_category_color={'Incorrect': 'red', 'Suspect': 'orange'},
             force_line_plot=False, labels=False, cbar_label=None, secondary_y=False,
             **kwargs):
        """
        Makes a timeseries plot. If subplots have not been added yet, an axis
        will be created assuming that there is only going to be one plot.

        Parameters
        ----------
        field : str
            The name of the field to plot.
        dsname : None or str
            If there is more than one datastream in the display object the
            name of the datastream needs to be specified. If set to None and
            there is only one datastream ACT will use the sole datastream
            in the object.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        cmap : matplotlib colormap
            The colormap to use.
        set_title : str
            The title for the plot.
        add_nan : bool
            Set to True to fill in data gaps with NaNs.
        day_night_background : bool
            Set to True to fill in a color coded background.
            according to the time of day.
        abs_limits : tuple or list
            Sets the bounds on plot limits even if data values exceed
            those limits. Set to (ymin,ymax). Use None if only setting
            minimum or maximum limit, i.e. (22., None).
        time_rng : tuple or list
            List or tuple with (min, max) values to set the x-axis range
            limits.
        use_var_for_y : str
            Set this to the name of a data variable in the Dataset to use as
            the y-axis variable instead of the default dimension. Useful for
            instances where data has an index-based dimension instead of a
            height-based dimension. If shapes of arrays do not match it will
            automatically revert back to the original ydata.
        assessment_overplot : boolean
            Option to overplot quality control colored symbols over plotted
            data using flag_assessment categories.
        assessment_overplot_category : dict
            Lookup to categorize assessments into groups. This allows using
            multiple terms for the same quality control level of failure.
            Also allows adding more to the defaults.
        assessment_overplot_category_color : dict
            Lookup to match overplot category color to assessment grouping.
        force_line_plot : boolean
            Option to plot 2D data as 1D line plots.
        labels : boolean or list
            Option to overwrite the legend labels. Must have same dimensions as
            number of lines plotted.
        cbar_label : str
            Option to overwrite default colorbar label.
        secondary_y : boolean
            Option to plot on secondary y axis.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.plot` (1D timeseries) or
            :func:`plt.pcolormesh` (2D timeseries).

        Returns
        -------
        ax : matplotlib axis handle
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

        if cbar_label is None:
            cbar_default = ytitle
        if len(dim) > 1:
            if use_var_for_y is None:
                ydata = self._arm[dsname][dim[1]]
            else:
                ydata = self._arm[dsname][use_var_for_y]
                ydata_dim1 = self._arm[dsname][dim[1]]
                if np.shape(ydata) != np.shape(ydata_dim1):
                    ydata = ydata_dim1
            units = ytitle
            if 'units' in ydata.attrs.keys():
                units = ydata.attrs['units']
                ytitle = ''.join(['(', units, ')'])
            else:
                units = ''
                ytitle = dim[1]

            # Create labels if 2d as 1d
            if force_line_plot is True:
                if labels is True:
                    labels = [' '.join([str(d), units]) for d in ydata.values]
                ydata = None
        else:
            ydata = None

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        # Set up secondary y axis if requested
        if secondary_y is False:
            ax = self.axes[subplot_index]
        else:
            ax = self.axes[subplot_index].twinx()

        if ydata is None:
            if day_night_background is True:
                self.day_night_background(subplot_index=subplot_index,
                                          dsname=dsname)

            # If limiting data being plotted use masked arrays
            # Need to do it this way because of autoscale() method
            if any(abs_limits):
                temp_data = np.ma.masked_invalid(data.values)
                if abs_limits[0] is not None and abs_limits[1] is not None:
                    temp_data = np.ma.masked_outside(
                        temp_data, abs_limits[0], abs_limits[1])
                elif abs_limits[0] is not None and abs_limits[1] is None:
                    temp_data = np.ma.masked_less_equal(
                        temp_data, abs_limits[0])
                elif abs_limits[0] is None and abs_limits[1] is not None:
                    temp_data = np.ma.masked_greater_equal(
                        temp_data, abs_limits[1])
                lines = ax.plot(xdata, temp_data, '.', **kwargs)

                # Overplot failing data if requested
                if assessment_overplot:
                    for assessment, categories in assessment_overplot_category.items():
                        flag_data = self._arm[dsname].qcfilter.get_masked_data(
                            field, rm_assessments=categories, return_inverse=True)
                        flag_data.mask = np.logical_or(flag_data.mask, temp_data.mask)
                        if any(np.invert(flag_data.mask)) and any(np.isfinite(flag_data)):
                            # qc_ax = self.axes[subplot_index].plot(
                            #    xdata, flag_data, marker='*', linestyle='',
                            #    color=assessment_overplot_category_color[assessment],
                            #    label=assessment)
                            # self.axes[subplot_index].legend(qc_ax, [assessment])
                            ax.legend()

            else:
                lines = ax.plot(xdata, data, '.', **kwargs)

                # Overplot failing data if requested
                if assessment_overplot:
                    for assessment, categories in assessment_overplot_category.items():
                        flag_data = self._arm[dsname].qcfilter.get_masked_data(
                            field, rm_assessments=categories, return_inverse=True)
                        if any(np.invert(flag_data.mask)) and any(np.isfinite(flag_data)):
                            # qc_ax = self.axes[subplot_index].plot(
                            #    xdata, flag_data, marker='*', linestyle='',
                            #    color=assessment_overplot_category_color[assessment],
                            #    label=assessment)
                            # self.axes[subplot_index].legend(qc_ax, [assessment])
                            ax.legend()

            # Add legend if labels are available
            if isinstance(labels, list):
                ax.legend(lines, labels)

        else:
            # Add in nans to ensure the data are not streaking
            if add_nan is True:
                xdata, data = data_utils.add_in_nan(xdata, data)
            mesh = ax.pcolormesh(xdata, ydata, data.transpose(),
                                 cmap=cmap, edgecolors='face', **kwargs)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, field, 'on',
                                 dt_utils.numpy_to_arm_date(
                                     self._arm[dsname].time.values[0])])

        if secondary_y is False:
            ax.set_title(set_title)

        # Set YTitle
        ax.set_ylabel(ytitle)

        # Set X Limit - We want the same time axes for all subplots
        if not hasattr(self, 'time_rng'):
            if time_rng is not None:
                self.time_rng = list(time_rng)
            else:
                self.time_rng = [xdata.min().values, xdata.max().values]

        self.set_xrng(self.time_rng, subplot_index)

        # Set Y Limit
        if hasattr(self, 'yrng'):
            # Make sure that the yrng is not just the default
            if ydata is None:
                if any(abs_limits):
                    our_data = temp_data
                else:
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

            # Check if current range is outside of new range an only set
            # values that work for all data plotted.
            current_yrng = ax.get_ylim()

            if yrng[0] > current_yrng[0]:
                yrng[0] = current_yrng[0]
            if yrng[1] < current_yrng[1]:
                yrng[1] = current_yrng[1]

            # Set y range the normal way if not secondary y
            # If secondary, just use set_ylim
            if secondary_y is False:
                self.set_yrng(yrng, subplot_index)
            else:
                ax.set_ylim(yrng)

        # Set X Format
        if len(subplot_index) == 1:
            days = (self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0])
        else:
            days = (self.xrng[subplot_index[0], subplot_index[1], 1] -
                    self.xrng[subplot_index[0], subplot_index[1], 0])

        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)

        # Set X format - We want the same time axes for all subplots
        if not hasattr(self, 'time_fmt'):
            self.time_fmt = myFmt

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            ax.set_xlabel('Time [UTC]')

        if ydata is not None:
            if cbar_label is None:
                self.add_colorbar(mesh, title=cbar_default, subplot_index=subplot_index)
            else:
                self.add_colorbar(mesh, title=''.join(['(', cbar_label, ')']),
                                  subplot_index=subplot_index)

        return ax

    def plot_barbs_from_spd_dir(self, dir_field, spd_field, pres_field=None,
                                dsname=None, **kwargs):
        """
        This procedure will make a wind barb plot timeseries.
        If a pressure field is given and the wind fields are 1D, which, for
        example, would occur if one wants to plot a timeseries of
        rawinsonde data, then a time-height cross section of
        winds will be made.

        Note: This procedure calls plot_barbs_from_u_v and will take in the
        same keyword arguments as that procedure.

        Parameters
        ----------
        dir_field : str
            The name of the field specifying the wind direction in degrees.
            0 degrees is defined to be north and increases clockwise like
            what is used in standard meteorological notation.
        spd_field : str
            The name of the field specifying the wind speed in m/s.
        pres_field : str
            The name of the field specifying pressure or height. If using
            height coordinates, then we recommend setting invert_y_axis
            to False.
        dsname : str
            The name of the datastream to plot. Setting to None will make
            ACT attempt to autodetect this.
        kwargs : dict
            Any additional keyword arguments will be passed into
            :func:`act.plotting.TimeSeriesDisplay.plot_barbs_from_u_and_v`.

        Returns
        -------
        the_ax : matplotlib axis handle
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
                            num_barbs_x=20, num_barbs_y=20,
                            use_var_for_y=None, **kwargs):
        """
        This function will plot a wind barb timeseries from u and v wind
        data. If pres_field is given, a time-height series will be plotted
        from 1-D wind data.

        Parameters
        ----------
        u_field : str
            The name of the field containing the U component of the wind.
        v_field : str
            The name of the field containing the V component of the wind.
        pres_field : str or None
            The name of the field containing the pressure or height. Set
            to None to not use this.
        dsname : str or None
            The name of the datastream to plot. Setting to None will make
            ACT automatically try to determine this.
        subplot_index : 2-tuple
            The index of the subplot to make the plot on.
        set_title : str or None
            The title of the plot.
        day_night_background : bool
            Set to True to plot a day/night background.
        invert_y_axis : bool
            Set to True to invert the y axis (i.e. for plotting pressure as
            the height coordinate).
        num_barbs_x : int
            The number of wind barbs to plot in the x axis.
        num_barbs_y : int
            The number of wind barbs to plot in the y axis.
        cmap : matplotlib.colors.LinearSegmentedColormap
            A color map to use with wind barbs. If this is set the plt.barbs
            routine will be passed the C parameter scaled as sqrt of sum of the
            squares and used with the passed in color map. A colorbar will also
            be added. Setting the limits of the colorbar can be done with 'clim'.
            Setting this changes the wind barbs from black to colors.
        use_var_for_y : str
            Set this to the name of a data variable in the Dataset to use as the
            y-axis variable instead of the default dimension. Useful for instances
            where data has an index-based dimension instead of a height-based
            dimension. If shapes of arrays do not match it will automatically
            revert back to the original ydata.
        **kwargs : keyword arguments
            Additional keyword arguments will be passed into plt.barbs.

        Returns
        -------
        ax : matplotlib axis handle
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
        if barb_step_x == 0:
            barb_step_x = 1
        if len(dim) > 1 and pres_field is None:
            if use_var_for_y is None:
                ydata = self._arm[dsname][dim[1]]
            else:
                ydata = self._arm[dsname][use_var_for_y]
                ydata_dim1 = self._arm[dsname][dim[1]]
                if np.shape(ydata) != np.shape(ydata_dim1):
                    ydata = ydata_dim1
            ytitle = ''.join(['(', ydata.attrs['units'], ')'])
            units = ytitle
            num_y = ydata.shape[0]
            barb_step_y = round(num_y / num_barbs_y)
            if barb_step_y == 0:
                barb_step_y = 1
            xdata, ydata = np.meshgrid(xdata, ydata, indexing='ij')
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
            if 'cmap' in kwargs.keys():
                map_color = np.sqrt(np.power(u[::barb_step_x], 2) +
                                    np.power(v[::barb_step_x], 2))
                map_color[np.isnan(map_color)] = 0
                ax = self.axes[subplot_index].barbs(xdata[::barb_step_x],
                                                    ydata[::barb_step_x],
                                                    u[::barb_step_x],
                                                    v[::barb_step_x], map_color,
                                                    **kwargs)
                plt.colorbar(ax, ax=[self.axes[subplot_index]],
                             label='Wind Speed (' +
                             self._arm[dsname][u_field].attrs['units'] + ')')

            else:
                self.axes[subplot_index].barbs(xdata[::barb_step_x],
                                               ydata[::barb_step_x],
                                               u[::barb_step_x],
                                               v[::barb_step_x],
                                               **kwargs)
            self.axes[subplot_index].set_yticks([])

        else:
            if 'cmap' in kwargs.keys():
                map_color = np.sqrt(np.power(u[::barb_step_x, ::barb_step_y], 2) +
                                    np.power(v[::barb_step_x, ::barb_step_y], 2))
                map_color[np.isnan(map_color)] = 0

                ax = self.axes[subplot_index].barbs(
                    xdata[::barb_step_x, ::barb_step_y],
                    ydata[::barb_step_x, ::barb_step_y],
                    u[::barb_step_x, ::barb_step_y],
                    v[::barb_step_x, ::barb_step_y], map_color,
                    **kwargs)
                plt.colorbar(ax, ax=[self.axes[subplot_index]],
                             label='Wind Speed (' +
                             self._arm[dsname][u_field].attrs['units'] + ')')
            else:
                ax = self.axes[subplot_index].barbs(
                    xdata[::barb_step_x, ::barb_step_y],
                    ydata[::barb_step_x, ::barb_step_y],
                    u[::barb_step_x, ::barb_step_y],
                    v[::barb_step_x, ::barb_step_y],
                    **kwargs)

        if day_night_background is True:
            self.day_night_background(subplot_index=subplot_index, dsname=dsname)

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
            num_y_levels=20, invert_y_axis=True, cbar_label=None,
            **kwargs):
        """
        This will plot a time-height cross section from 1D datasets using
        nearest neighbor interpolation on a regular time by height grid.
        All that is needed are a data variable and a height variable.

        Parameters
        ----------
        data_field : str
            The name of the field to plot.
        pres_field : str
            The name of the height or pressure field to plot.
        dsname : str or None
            The name of the datastream to plot
        subplot_index : 2-tuple
            The index of the subplot to create the plot on.
        set_title : str or None
            The title of the plot.
        day_night_background : bool
            Set to true to plot the day/night background.
        num_time_periods : int
            Set to determine how many time periods. Setting to None
            will do one time period per day.
        num_y_levels : int
            The number of levels in the y axis to use.
        invert_y_axis : bool
            Set to true to invert the y-axis (recommended for
            pressure coordinates).
        cbar_label : str
            Option to overwrite default colorbar label.
        **kwargs : keyword arguments
            Additional keyword arguments will be passed
            into :func:`plt.pcolormesh`

        Returns
        -------
        ax : matplotlib axis handle
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
            self.day_night_background(subplot_index=subplot_index, dsname=dsname)

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
            if cbar_label is None:
                self.add_colorbar(mesh, title=units, subplot_index=subplot_index)
            else:
                self.add_colorbar(mesh, title=cbar_label, subplot_index=subplot_index)
        myFmt = common.get_date_format(days)
        self.axes[subplot_index].xaxis.set_major_formatter(myFmt)

        return self.axes[subplot_index]

    def time_height_scatter(
            self, data_field=None, dsname=None, cmap='rainbow',
            alt_label=None, alt_field='alt', cb_label=None, **kwargs):
        """
        Create a time series plot of altitude and data variable with
        color also indicating value with a color bar. The Color bar is
        positioned to serve both as the indicator of the color intensity
        and the second y-axis.

        Parameters
        ----------
        data_field : str
            Name of data field in the object to plot on second y-axis.
        height_field : str
            Name of height field in the object to plot on first y-axis.
        dsname : str or None
            The name of the datastream to plot.
        cmap : str
            Colorbar color map to use.
        alt_label : str
            Altitude first y-axis label to use. If None, will try to use
            long_name and units.
        alt_field : str
            Label for field in the object to plot on first y-axis.
        cb_label : str
            Colorbar label to use. If not set will try to use
            long_name and units.
        **kwargs : keyword arguments
            Any other keyword arguments that will be passed
            into TimeSeriesDisplay.plot module when the figure
            is made.

        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Get data and dimensions
        data = self._arm[dsname][data_field]
        altitude = self._arm[dsname][alt_field]
        dim = list(self._arm[dsname][data_field].dims)
        xdata = self._arm[dsname][dim[0]]

        if alt_label is None:
            try:
                alt_label = (altitude.attrs['long_name'] +
                             ''.join([' (', altitude.attrs['units'], ')']))
            except KeyError:
                alt_label = alt_field

        if cb_label is None:
            try:
                cb_label = (data.attrs['long_name'] +
                            ''.join([' (', data.attrs['units'], ')']))
            except KeyError:
                cb_label = data_field

        colorbar_map = plt.cm.get_cmap(cmap)
        self.fig.subplots_adjust(left=0.1, right=0.86,
                                 bottom=0.16, top=0.91)
        ax1 = self.plot(alt_field, color='black', **kwargs)
        ax1.set_ylabel(alt_label)
        ax2 = ax1.twinx()
        sc = ax2.scatter(xdata.values, data.values, c=data.values,
                         marker='.', cmap=colorbar_map)
        cbaxes = self.fig.add_axes(
            [self.fig.subplotpars.right + 0.02, self.fig.subplotpars.bottom,
             0.02, self.fig.subplotpars.top - self.fig.subplotpars.bottom])
        cbar = plt.colorbar(sc, cax=cbaxes)
        ax2.set_ylim(cbar.mappable.get_clim())
        cbar.ax.set_ylabel(cb_label)
        ax2.set_yticklabels([])

        return self.axes[0]

    def qc_flag_block_plot(
            self, data_field=None, dsname=None,
            subplot_index=(0, ), time_rng=None, assesment_color=None,
            edgecolor='face', **kwargs):
        """
        Create a time series plot of embedded quality control values
        using broken barh plotting.

        Parameters
        ----------
        data_field : str
            Name of data field in the object to plot corresponding quality
            control.
        dsname : None or str
            If there is more than one datastream in the display object the
            name of the datastream needs to be specified. If set to None and
            there is only one datastream ACT will use the sole datastream
            in the object.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        time_rng : tuple or list
            List or tuple with (min, max) values to set the x-axis range limits.
        assesment_color : dict
            Dictionary lookup to override default assessment to color. Make sure
            assessment work is correctly set with case syntax.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.broken_barh`.

        """
        # Color to plot associated with assessment.
        color_lookup = {'Bad': 'red',
                        'Incorrect': 'red',
                        'Indeterminate': 'orange',
                        'Suspect': 'orange',
                        'Missing': 'darkgray'}

        if assesment_color is not None:
            for asses, color in assesment_color.items():
                color_lookup[asses] = color
                if asses == 'Incorrect':
                    color_lookup['Bad'] = color
                if asses == 'Suspect':
                    color_lookup['Indeterminate'] = color

        # Set up list of test names to use for missing values
        missing_val_long_names = ['Value.* equal to missing_value*',
                                  'Value.* set to missing_value*']

        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]

        # Set up or get current plot figure
        if self.fig is None:
            self.fig = plt.figure()

        # Set up or get current axes
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        ax = self.axes[subplot_index]

        # Set X Limit - We want the same time axes for all subplots
        data = self._arm[dsname][data_field]
        dim = list(self._arm[dsname][data_field].dims)
        xdata = self._arm[dsname][dim[0]]

        # Get data and attributes
        qc_data_field = self._arm[dsname].qcfilter.check_for_ancillary_qc(data_field)
        flag_masks = self._arm[dsname][qc_data_field].attrs['flag_masks']
        flag_meanings = self._arm[dsname][qc_data_field].attrs['flag_meanings']
        flag_assessments = self._arm[dsname][qc_data_field].attrs['flag_assessments']

        # Get time ranges for green blocks
        time_delta = determine_time_delta(xdata.values)
        barh_list_green = reduce_time_ranges(xdata.values, time_delta=time_delta,
                                             broken_barh=True)

        # Set background to gray indicating not available data
        ax.set_facecolor('dimgray')

        test_nums = []
        for ii, assess in enumerate(flag_assessments):
            # Plot green data first.
            ax.broken_barh(barh_list_green, (ii, ii + 1), facecolors='green',
                           edgecolor=edgecolor, **kwargs)
            # Get test number from flag_mask bitpacked number
            test_nums.append(parse_bit(flag_masks[ii]))
            # Get masked array data to use mask for finding if/where test is set
            data = self._arm[dsname].qcfilter.get_masked_data(
                data_field, rm_tests=test_nums[-1])
            if np.any(data.mask):
                # Get time ranges from time and masked data
                barh_list = reduce_time_ranges(xdata.values[data.mask],
                                               time_delta=time_delta,
                                               broken_barh=True)
                # Check if the bit set is indicating missing data. If so change
                # to different plotting color than what is in flag_assessments.
                for val in missing_val_long_names:
                    if re_search(val, flag_meanings[ii]):
                        assess = "Missing"
                        break
                # Lay down blocks of tripped tests using correct color
                ax.broken_barh(barh_list, (ii, ii + 1),
                               facecolors=color_lookup[assess],
                               edgecolor=edgecolor, **kwargs)

            # Add test description to plot.
            ax.text(xdata.values[0], ii + 0.5, ' ' + flag_meanings[ii], va='center')

        # Change y ticks to test number
        plt.yticks([ii + 0.5 for ii in range(0, len(test_nums))],
                   labels=['Test ' + str(ii[0]) for ii in test_nums])
        # Set ylimit to number of tests plotted
        ax.set_ylim(0, len(flag_assessments))
        # Set X Limit - We want the same time axes for all subplots
        if not hasattr(self, 'time_rng'):
            if time_rng is not None:
                self.time_rng = list(time_rng)
            else:
                self.time_rng = [xdata.min().values, xdata.max().values]

        self.set_xrng(self.time_rng, subplot_index)

        # Get X format - We want the same time axes for all subplots
        if hasattr(self, 'time_fmt'):
            ax.xaxis.set_major_formatter(self.time_fmt)
        else:
            # Set X Format
            if len(subplot_index) == 1:
                days = (self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0])
            else:
                days = (self.xrng[subplot_index[0], subplot_index[1], 1] -
                        self.xrng[subplot_index[0], subplot_index[1], 0])

            myFmt = common.get_date_format(days)
            ax.xaxis.set_major_formatter(myFmt)
            self.time_fmt = myFmt

        return self.axes[subplot_index]

    def fill_between(self, field, dsname=None, subplot_index=(0, ),
                     set_title=None, secondary_y=False, **kwargs):
        """
        Makes a fill_between plot, based on matplotlib

        Parameters
        ----------
        field : str
            The name of the field to plot.
        dsname : None or str
            If there is more than one datastream in the display object the
            name of the datastream needs to be specified. If set to None and
            there is only one datastream ACT will use the sole datastream
            in the object.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        set_title : str
            The title for the plot.
        secondary_y : boolean
            Option to indicate if the data should be plotted on second y-axis.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.plot` (1D timeseries) or
            :func:`plt.pcolormesh` (2D timeseries).

        Returns
        -------
        ax : matplotlib axis handle
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

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        # Set ax to appropriate axis
        if secondary_y is False:
            ax = self.axes[subplot_index]
        else:
            ax = self.axes[subplot_index].twinx()

        ax.fill_between(xdata.values, data, **kwargs)

        # Set X Format
        if len(subplot_index) == 1:
            days = (self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0])
        else:
            days = (self.xrng[subplot_index[0], subplot_index[1], 1] -
                    self.xrng[subplot_index[0], subplot_index[1], 0])

        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)

        # Set X format - We want the same time axes for all subplots
        if not hasattr(self, 'time_fmt'):
            self.time_fmt = myFmt

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            self.axes[subplot_index].set_xlabel('Time [UTC]')

        # Set YTitle
        ax.set_ylabel(ytitle)

        # Set Title
        if set_title is None:
            set_title = ' '.join([dsname, field, 'on',
                                 dt_utils.numpy_to_arm_date(
                                     self._arm[dsname].time.values[0])])
        if secondary_y is False:
            ax.set_title(set_title)

        return self.axes[subplot_index]
