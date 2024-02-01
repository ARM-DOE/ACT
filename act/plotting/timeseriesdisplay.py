"""
Stores the class for TimeSeriesDisplay.

"""

import datetime as dt
import textwrap
import warnings
from copy import deepcopy
from re import search, search as re_search

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mplcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import NearestNDInterpolator

from ..qc.qcfilter import parse_bit
from ..utils import data_utils, datetime_utils as dt_utils
from ..utils.datetime_utils import determine_time_delta, reduce_time_ranges
from ..utils.geo_utils import get_sunrise_sunset_noon
from . import common
from .plot import Display


class TimeSeriesDisplay(Display):
    """
    This subclass contains routines that are specific to plotting
    time series plots from data. It is inherited from Display and therefore
    contains all of Display's attributes and methods.

    Examples
    --------

    To create a TimeSeriesDisplay with 3 rows, simply do:

    .. code-block:: python

        ds = act.io.read_arm_netcdf(the_file)
        disp = act.plotting.TimeSeriesDisplay(ds, subplot_shape=(3,), figsize=(15, 5))

    The TimeSeriesDisplay constructor takes in the same keyword arguments as
    plt.subplots. For more information on the plt.subplots keyword arguments,
    see the `matplotlib documentation
    <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html>`_.
    If no subplot_shape is provided, then no figure or axis will be created
    until add_subplots or plots is called.

    """

    def __init__(self, ds, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(ds, subplot_shape, ds_name, **kwargs)

    def day_night_background(self, dsname=None, subplot_index=(0,)):
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

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream to derive the '
                + 'information needed for the day and night '
                + 'background when 2 or more datasets are in '
                + 'the display object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Get File Dates
        try:
            file_dates = self._ds[dsname].attrs['_file_dates']
        except KeyError:
            file_dates = []
        if len(file_dates) == 0:
            sdate = dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0])
            edate = dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[-1])
            file_dates = [sdate, edate]

        all_dates = dt_utils.dates_between(file_dates[0], file_dates[-1])

        if self.axes is None:
            raise RuntimeError('day_night_background requires the plot to ' 'be displayed.')

        ax = self.axes[subplot_index]

        # Find variable names for latitude and longitude
        variables = list(self._ds[dsname].data_vars)
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
                    if self._ds[dsname][var].attrs['standard_name'] == 'latitude':
                        lat_name = var
                except KeyError:
                    pass

                try:
                    if self._ds[dsname][var].attrs['standard_name'] == 'longitude':
                        lon_name = var
                except KeyError:
                    pass

                if lat_name is not None and lon_name is not None:
                    break

        if lat_name is None or lon_name is None:
            return

        # Extract latitude and longitude scalar from variable. If variable is a vector look
        # for first non-Nan value.
        lat_lon_list = [np.nan, np.nan]
        for ii, var_name in enumerate([lat_name, lon_name]):
            try:
                values = self._ds[dsname][var_name].values
                if values.size == 1:
                    lat_lon_list[ii] = float(values)
                else:
                    # Look for non-NaN values to use for latitude locaiton. If not found use first value.
                    index = np.where(np.isfinite(values))[0]
                    if index.size == 0:
                        lat_lon_list[ii] = float(values[0])
                    else:
                        lat_lon_list[ii] = float(values[index[0]])
            except AttributeError:
                pass

        for value, name in zip(lat_lon_list, ['Latitude', 'Longitude']):
            if not np.isfinite(value):
                warnings.warn(
                    f"{name} value in dataset equal to '{value}' is not finite. ", RuntimeWarning
                )
                return

        lat = lat_lon_list[0]
        lon = lat_lon_list[1]

        lat_range = [-90, 90]
        if not (lat_range[0] <= lat <= lat_range[1]):
            warnings.warn(
                f"Latitude value in dataset of '{lat}' not within acceptable "
                f'range of {lat_range[0]} <= latitude <= {lat_range[1]}. ',
                RuntimeWarning,
            )
            return

        lon_range = [-180, 180]
        if not (lon_range[0] <= lon <= lon_range[1]):
            warnings.warn(
                f"Longitude value in dataset of '{lon}' not within acceptable "
                f'range of {lon_range[0]} <= longitude <= {lon_range[1]}. ',
                RuntimeWarning,
            )
            return

        # Initialize the plot to a gray background for total darkness
        rect = ax.patch
        rect.set_facecolor('0.85')

        # Get date ranges to plot
        plot_dates = []
        for f in all_dates:
            for ii in [-1, 0, 1]:
                plot_dates.append(f + dt.timedelta(days=ii))

        # Get sunrise, sunset and noon times
        sunrise, sunset, noon = get_sunrise_sunset_noon(lat, lon, plot_dates)

        # Plot daylight
        for ii in range(0, len(sunrise)):
            ax.axvspan(sunrise[ii], sunset[ii], facecolor='#FFFFCC', zorder=0)

        # Plot noon line
        for ii in noon:
            ax.axvline(x=ii, linestyle='--', color='y', zorder=1)

    def set_xrng(self, xrng, subplot_index=(0, 0)):
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

        # If the xlim is set to the same value for range it will throw a warning
        # This is to catch that and expand the range so we avoid the warning.
        if xrng[0] == xrng[1]:
            if isinstance(xrng[0], np.datetime64):
                print(
                    f'\nAttempting to set xlim range to single value {xrng[0]}. '
                    'Expanding range by 2 seconds.\n'
                )
                xrng[0] -= np.timedelta64(1, 's')
                xrng[1] += np.timedelta64(1, 's')
            elif isinstance(xrng[0], dt.datetime):
                print(
                    f'\nAttempting to set xlim range to single value {xrng[0]}. '
                    'Expanding range by 2 seconds.\n'
                )
                xrng[0] -= dt.timedelta(seconds=1)
                xrng[1] += dt.timedelta(seconds=1)
        self.axes[subplot_index].set_xlim(xrng)

        # Make sure that the xrng value is a numpy array not pandas
        if isinstance(xrng[0], pd.Timestamp):
            xrng = [x.to_numpy() for x in xrng if isinstance(x, pd.Timestamp)]

        # Make sure that the xrng value is a numpy array not datetime.datetime
        if isinstance(xrng[0], dt.datetime):
            xrng = [np.datetime64(x) for x in xrng if isinstance(x, dt.datetime)]

        if len(subplot_index) < 2:
            self.xrng[subplot_index, 0] = xrng[0].astype('datetime64[D]').astype(float)
            self.xrng[subplot_index, 1] = xrng[1].astype('datetime64[D]').astype(float)
        else:
            self.xrng[subplot_index][0] = xrng[0].astype('datetime64[D]').astype(float)
            self.xrng[subplot_index][1] = xrng[1].astype('datetime64[D]').astype(float)

    def set_yrng(self, yrng, subplot_index=(0,), match_axes_ylimits=False):
        """
        Sets the y range of the plot.

        Parameters
        ----------
        yrng : 2 number array
            The y limits of the plot.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the y range of. This is
            ignored if match_axes_ylimits is True.
        match_axes_ylimits : boolean
            If True, all axes in the display object will have matching
            provided ylims. Default is False. This is especially useful
            when utilizing a groupby display with many axes.

        """
        if self.axes is None:
            raise RuntimeError('set_yrng requires the plot to be displayed.')

        if not hasattr(self, 'yrng') and len(self.axes.shape) == 2:
            self.yrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2))
        elif not hasattr(self, 'yrng') and len(self.axes.shape) == 1:
            self.yrng = np.zeros((self.axes.shape[0], 2))

        if yrng[0] == yrng[1]:
            yrng[1] = yrng[1] + 1

        # Sets all axes ylims to the same values.
        if match_axes_ylimits:
            for i in range(self.axes.shape[0]):
                for j in range(self.axes.shape[1]):
                    self.axes[i, j].set_ylim(yrng)
        else:
            self.axes[subplot_index].set_ylim(yrng)

        try:
            self.yrng[subplot_index, :] = yrng
        except IndexError:
            self.yrng[subplot_index] = yrng

    def plot(
        self,
        field,
        dsname=None,
        subplot_index=(0,),
        cmap=None,
        set_title=None,
        add_nan=False,
        day_night_background=False,
        invert_y_axis=False,
        abs_limits=(None, None),
        time_rng=None,
        y_rng=None,
        use_var_for_y=None,
        set_shading='auto',
        assessment_overplot=False,
        overplot_marker='.',
        overplot_behind=False,
        overplot_markersize=6,
        assessment_overplot_category={
            'Incorrect': ['Bad', 'Incorrect'],
            'Suspect': ['Indeterminate', 'Suspect'],
        },
        assessment_overplot_category_color={'Incorrect': 'red', 'Suspect': 'orange'},
        force_line_plot=False,
        labels=False,
        cbar_label=None,
        cbar_h_adjust=None,
        y_axis_flag_meanings=False,
        colorbar_labels=None,
        cb_friendly=False,
        match_line_label_color=False,
        **kwargs,
    ):
        """
        Makes a timeseries plot. If subplots have not been added yet, an axis
        will be created assuming that there is only going to be one plot.

        If plotting a high data volume 2D dataset, it may take some time to plot.
        In order to speed up your plot creation, please resample your data to a
        lower resolution dataset.

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
        y_rng : tuple or list
            List or tuple with (min, max) values to set the y-axis range
        use_var_for_y : str
            Set this to the name of a data variable in the Dataset to use as
            the y-axis variable instead of the default dimension. Useful for
            instances where data has an index-based dimension instead of a
            height-based dimension. If shapes of arrays do not match it will
            automatically revert back to the original ydata.
        set_shading : string
            Option to to set the matplotlib.pcolormesh shading parameter.
            Default to 'auto'
        assessment_overplot : boolean
            Option to overplot quality control colored symbols over plotted
            data using flag_assessment categories.
        overplot_marker : str
            Marker to use for overplot symbol.
        overplot_behind : bool
            Place the overplot marker behind the data point.
        overplot_markersize : float or int
            Size of overplot marker. If overplot_behind or force_line_plot
            are set the marker size will be double overplot_markersize so
            the color is visible.
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
        cbar_h_adjust : float
            Option to adjust location of colorbar horizontally. Positive values
            move to right negative values move to left.
        y_axis_flag_meanings : boolean or int
            When set to True and plotting state variable with flag_values and
            flag_meanings attributes will replace y axis numerical values
            with flag_meanings value. Set to a positive number larger than 1
            to indicate maximum word length to use. If text is longer that the
            value and has space characters will split text over multiple lines.
        colorbar_labels : dict
            A dictionary containing values for plotting a 2D array of state variables.
            The dictionary uses data values as keys and a dictionary containing keys
            'text' and 'color' for each data value to plot.

            Example:
                {0: {'text': 'Clear sky', 'color': 'white'},
                1: {'text': 'Liquid', 'color': 'green'},
                2: {'text': 'Ice', 'color': 'blue'},
                3: {'text': 'Mixed phase', 'color': 'purple'}}
        cb_friendly : boolean
            Set to true if you want to use the integrated colorblind friendly
            colors for green/red based on the Homeyer colormap.
        match_line_label_color : boolean
            Will set the y label to match the line color in the plot. This
            will only work if the time series plot is a line plot.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.plot` (1D timeseries) or
            :func:`plt.pcolormesh` (2D timeseries).

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        if y_axis_flag_meanings:
            kwargs['linestyle'] = ''

        if cb_friendly:
            cmap = 'HomeyerRainbow'
            assessment_overplot_category_color['Bad'] = (
                0.9285714285714286,
                0.7130901016453677,
                0.7130901016453677,
            )
            assessment_overplot_category_color['Incorrect'] = (
                0.9285714285714286,
                0.7130901016453677,
                0.7130901016453677,
            )
            assessment_overplot_category_color['Not Failing'] = (
                (0.0, 0.4240129715562796, 0.4240129715562796),
            )
            assessment_overplot_category_color['Acceptable'] = (
                (0.0, 0.4240129715562796, 0.4240129715562796),
            )

        # Get data and dimensions
        data = self._ds[dsname][field]
        dim = list(self._ds[dsname][field].dims)
        xdata = self._ds[dsname][dim[0]]

        if 'units' in data.attrs:
            ytitle = ''.join(['(', data.attrs['units'], ')'])
        else:
            ytitle = field

        if cbar_label is None:
            cbar_default = ytitle
        if len(dim) > 1:
            if use_var_for_y is None:
                ydata = self._ds[dsname][dim[1]]
            else:
                ydata = self._ds[dsname][use_var_for_y]
                ydata_dim1 = self._ds[dsname][dim[1]]
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
                if 'units' in data.attrs.keys():
                    units = data.attrs['units']
                    ytitle = ''.join(['(', units, ')'])
                else:
                    units = ''
                    ytitle = dim[1]
                ydata = None
        else:
            ydata = None

        # Get the current plotting axis
        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        ax = self.axes[subplot_index]

        if colorbar_labels is not None:
            flag_values = list(colorbar_labels.keys())
            flag_meanings = [value['text'] for key, value in colorbar_labels.items()]
            cbar_colors = [value['color'] for key, value in colorbar_labels.items()]
            cmap = mpl.colors.ListedColormap(cbar_colors)
            for ii, flag_meaning in enumerate(flag_meanings):
                if len(flag_meaning) > 20:
                    flag_meaning = textwrap.fill(flag_meaning, width=20)
                    flag_meanings[ii] = flag_meaning
        else:
            flag_values = None
            flag_meanings = None
            cbar_colors = None

        if ydata is None:
            # Add in nans to ensure the data does not connect the line.
            if add_nan is True:
                xdata, data = data_utils.add_in_nan(xdata, data)

            if day_night_background is True:
                self.day_night_background(subplot_index=subplot_index, dsname=dsname)

            # If limiting data being plotted use masked arrays
            # Need to do it this way because of autoscale() method
            if abs_limits[0] is not None and abs_limits[1] is not None:
                data = np.ma.masked_outside(data, abs_limits[0], abs_limits[1])
            elif abs_limits[0] is not None and abs_limits[1] is None:
                data = np.ma.masked_less_equal(data, abs_limits[0])
            elif abs_limits[0] is None and abs_limits[1] is not None:
                data = np.ma.masked_greater_equal(data, abs_limits[1])

            # Plot the data
            if 'marker' not in kwargs.keys():
                kwargs['marker'] = '.'

            lines = ax.plot(xdata, data, **kwargs)

            # Check if we need to call legend method after plotting. This is only
            # called when no assessment overplot is called.
            add_legend = False
            if 'label' in kwargs.keys():
                add_legend = True

            # Overplot failing data if requested
            if assessment_overplot:
                # If we are doing forced line plot from 2D data need to manage
                # legend lables. Will make arrays to hold labels of QC failing
                # because not set when labels not set.
                if not isinstance(labels, list) and add_legend is False:
                    labels = []
                    lines = []

                # For forced line plot need to plot QC behind point instead of
                # on top of point.
                zorder = None
                if force_line_plot or overplot_behind:
                    zorder = 0
                    overplot_markersize *= 2.0

                for assessment, categories in assessment_overplot_category.items():
                    flag_data = self._ds[dsname].qcfilter.get_masked_data(
                        field, rm_assessments=categories, return_inverse=True
                    )
                    if np.invert(flag_data.mask).any() and np.isfinite(flag_data).any():
                        try:
                            flag_data.mask = np.logical_or(data.mask, flag_data.mask)
                        except AttributeError:
                            pass
                        qc_ax = ax.plot(
                            xdata,
                            flag_data,
                            marker=overplot_marker,
                            linestyle='',
                            markersize=overplot_markersize,
                            color=assessment_overplot_category_color[assessment],
                            label=assessment,
                            zorder=zorder,
                        )
                        # If labels keyword is set need to add labels for calling legend
                        if isinstance(labels, list):
                            # If plotting forced_line_plot need to subset the Line2D object
                            # so we don't have more than one added to legend.
                            if len(qc_ax) > 1:
                                lines.extend(qc_ax[:1])
                            else:
                                lines.extend(qc_ax)
                            labels.append(assessment)
                        add_legend = True

            # Add legend if labels are available
            if isinstance(labels, list):
                ax.legend(lines, labels)
            elif add_legend:
                ax.legend()

            # Change y axis to text from flag_meanings if requested.
            if y_axis_flag_meanings:
                flag_meanings = self._ds[dsname][field].attrs['flag_meanings']
                flag_values = self._ds[dsname][field].attrs['flag_values']
                # If keyword is larger than 1 assume this is the maximum character length
                # desired and insert returns to wrap text.
                if y_axis_flag_meanings > 1:
                    for ii, flag_meaning in enumerate(flag_meanings):
                        if len(flag_meaning) > y_axis_flag_meanings:
                            flag_meaning = textwrap.fill(flag_meaning, width=y_axis_flag_meanings)
                            flag_meanings[ii] = flag_meaning

                ax.set_yticks(flag_values)
                ax.set_yticklabels(flag_meanings)

        else:
            # Add in nans to ensure the data are not streaking
            if add_nan is True:
                xdata, data = data_utils.add_in_nan(xdata, data)

            # Sets shading parameter to auto. Matplotlib will check deminsions.
            # If X,Y and C are same deminsions shading is set to nearest.
            # If X and Y deminsions are 1 greater than C shading is set to flat.
            if 'edgecolors' not in kwargs.keys():
                kwargs['edgecolors'] = 'face'
            mesh = ax.pcolormesh(
                np.asarray(xdata),
                ydata,
                data.transpose(),
                shading=set_shading,
                cmap=cmap,
                **kwargs,
            )

        # Set Title
        if set_title is None:
            if isinstance(self._ds[dsname].time.values[0], np.datetime64):
                set_title = ' '.join(
                    [
                        dsname,
                        field,
                        'on',
                        dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                    ]
                )
            else:
                date_result = search(r'\d{4}-\d{1,2}-\d{1,2}', self._ds[dsname].time.attrs['units'])
                if date_result is not None:
                    set_title = ' '.join([dsname, field, 'on', date_result.group(0)])
                else:
                    set_title = ' '.join([dsname, field])

        ax.set_title(set_title)

        # Set YTitle
        if not y_axis_flag_meanings:
            if match_line_label_color and len(ax.get_lines()) > 0:
                ax.set_ylabel(ytitle, color=ax.get_lines()[0].get_color())
            else:
                ax.set_ylabel(ytitle)

        # Set X Limit - We want the same time axes for all subplots
        if not hasattr(self, 'time_rng'):
            if time_rng is not None:
                self.time_rng = list(time_rng)
            else:
                self.time_rng = [xdata.min().values, xdata.max().values]

        self.set_xrng(self.time_rng, subplot_index)

        # Set Y Limit
        if y_rng is not None:
            self.set_yrng(y_rng)

        if hasattr(self, 'yrng'):
            # Make sure that the yrng is not just the default
            if ydata is None:
                if abs_limits[0] is not None or abs_limits[1] is not None:
                    our_data = data
                else:
                    our_data = data.values
            else:
                our_data = ydata

            finite = np.isfinite(our_data)
            # If finite is returned as DataArray or Dask array extract values.
            try:
                finite = finite.values
            except AttributeError:
                pass

            if finite.any():
                our_data = our_data[finite]
                if invert_y_axis is False:
                    yrng = [np.min(our_data), np.max(our_data)]
                else:
                    yrng = [np.max(our_data), np.min(our_data)]
            else:
                yrng = [0, 1]

            # Check if current range is outside of new range an only set
            # values that work for all data plotted.
            if isinstance(yrng[0], np.datetime64):
                yrng = mdates.datestr2num([str(yrng[0]), str(yrng[1])])

            current_yrng = ax.get_ylim()
            if invert_y_axis is False:
                if yrng[0] > current_yrng[0]:
                    yrng[0] = current_yrng[0]
                if yrng[1] < current_yrng[1]:
                    yrng[1] = current_yrng[1]
            else:
                if yrng[0] < current_yrng[0]:
                    yrng[0] = current_yrng[0]
                if yrng[1] > current_yrng[1]:
                    yrng[1] = current_yrng[1]

            self.set_yrng(yrng, subplot_index)

        # Set X Format
        if len(subplot_index) == 1:
            days = self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0]
        else:
            days = self.xrng[subplot_index][1] - self.xrng[subplot_index][0]

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
                cbar_title = cbar_default
            else:
                cbar_title = ''.join(['(', cbar_label, ')'])

            if colorbar_labels is not None:
                cbar_title = None
                cbar = self.add_colorbar(
                    mesh,
                    title=cbar_title,
                    subplot_index=subplot_index,
                    values=flag_values,
                    pad=cbar_h_adjust,
                )
                cbar.set_ticks(flag_values)
                cbar.set_ticklabels(flag_meanings)
                cbar.ax.tick_params(labelsize=10)

            else:
                self.add_colorbar(
                    mesh, title=cbar_title, subplot_index=subplot_index, pad=cbar_h_adjust
                )
        return ax

    def plot_barbs_from_spd_dir(
        self, speed_field, direction_field, pres_field=None, dsname=None, **kwargs
    ):
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
        speed_field : str
            The name of the field specifying the wind speed in m/s.
        direction_field : str
            The name of the field specifying the wind direction in degrees.
            0 degrees is defined to be north and increases clockwise like
            what is used in standard meteorological notation.
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

            sonde_ds = act.io.arm.read_arm_netcdf(
                act.tests.sample_files.EXAMPLE_TWP_SONDE_WILDCARD)
            BarbDisplay = act.plotting.TimeSeriesDisplay(
                {'sonde_darwin': sonde_ds}, figsize=(10,5))
            BarbDisplay.plot_barbs_from_spd_dir('deg', 'wspd', 'pres',
                                                num_barbs_x=20)

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Make temporary field called tempu, tempv
        spd = self._ds[dsname][speed_field]
        dir = self._ds[dsname][direction_field]
        tempu = -np.sin(np.deg2rad(dir)) * spd
        tempv = -np.cos(np.deg2rad(dir)) * spd
        self._ds[dsname]['temp_u'] = deepcopy(self._ds[dsname][speed_field])
        self._ds[dsname]['temp_v'] = deepcopy(self._ds[dsname][speed_field])
        self._ds[dsname]['temp_u'].values = tempu
        self._ds[dsname]['temp_v'].values = tempv
        the_ax = self.plot_barbs_from_u_v('temp_u', 'temp_v', pres_field, dsname, **kwargs)
        del self._ds[dsname]['temp_u'], self._ds[dsname]['temp_v']
        return the_ax

    def plot_barbs_from_u_v(
        self,
        u_field,
        v_field,
        pres_field=None,
        dsname=None,
        subplot_index=(0,),
        set_title=None,
        day_night_background=False,
        invert_y_axis=True,
        num_barbs_x=20,
        num_barbs_y=20,
        use_var_for_y=None,
        **kwargs,
    ):
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
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Get data and dimensions
        u = self._ds[dsname][u_field].values
        v = self._ds[dsname][v_field].values
        dim = list(self._ds[dsname][u_field].dims)
        xdata = self._ds[dsname][dim[0]].values
        num_x = xdata.shape[-1]
        barb_step_x = round(num_x / num_barbs_x)
        if barb_step_x == 0:
            barb_step_x = 1

        if len(dim) > 1 and pres_field is None:
            if use_var_for_y is None:
                ydata = self._ds[dsname][dim[1]]
            else:
                ydata = self._ds[dsname][use_var_for_y]
                ydata_dim1 = self._ds[dsname][dim[1]]
                if np.shape(ydata) != np.shape(ydata_dim1):
                    ydata = ydata_dim1
            if 'units' in ydata.attrs:
                units = ydata.attrs['units']
            else:
                units = ''
            ytitle = ''.join(['(', units, ')'])
            num_y = ydata.shape[0]
            barb_step_y = round(num_y / num_barbs_y)
            if barb_step_y == 0:
                barb_step_y = 1

            xdata, ydata = np.meshgrid(xdata, ydata, indexing='ij')
        elif pres_field is not None:
            # What we will do here is do a nearest-neighbor interpolation
            # for each member of the series. Coordinates are time, pressure
            pres = self._ds[dsname][pres_field]
            u_interp = NearestNDInterpolator((xdata, pres.values), u, rescale=True)
            v_interp = NearestNDInterpolator((xdata, pres.values), v, rescale=True)
            barb_step_x = 1
            barb_step_y = 1
            x_times = pd.date_range(xdata.min(), xdata.max(), periods=num_barbs_x)
            if num_barbs_y == 1:
                y_levels = pres.mean()
            else:
                y_levels = np.linspace(np.nanmin(pres), np.nanmax(pres), num_barbs_y)
            xdata, ydata = np.meshgrid(x_times, y_levels, indexing='ij')
            u = u_interp(xdata, ydata)
            v = v_interp(xdata, ydata)
            if 'units' in pres.attrs:
                units = pres.attrs['units']
            else:
                units = ''
            ytitle = ''.join(['(', units, ')'])
        else:
            ydata = None

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        # Set up or get current axes
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        ax = self.axes[subplot_index]

        if ydata is None:
            ydata = np.ones(xdata.shape)
            if 'cmap' in kwargs.keys():
                map_color = np.sqrt(np.power(u[::barb_step_x], 2) + np.power(v[::barb_step_x], 2))
                map_color[np.isnan(map_color)] = 0
                barbs = ax.barbs(
                    xdata[::barb_step_x],
                    ydata[::barb_step_x],
                    u[::barb_step_x],
                    v[::barb_step_x],
                    map_color,
                    **kwargs,
                )
                plt.colorbar(
                    barbs,
                    ax=[ax],
                    label='Wind Speed (' + self._ds[dsname][u_field].attrs['units'] + ')',
                )

            else:
                ax.barbs(
                    xdata[::barb_step_x],
                    ydata[::barb_step_x],
                    u[::barb_step_x],
                    v[::barb_step_x],
                    **kwargs,
                )
            ax.set_yticks([])

        else:
            if 'cmap' in kwargs.keys():
                map_color = np.sqrt(
                    np.power(u[::barb_step_x, ::barb_step_y], 2)
                    + np.power(v[::barb_step_x, ::barb_step_y], 2)
                )
                map_color[np.isnan(map_color)] = 0
                barbs = ax.barbs(
                    xdata[::barb_step_x, ::barb_step_y],
                    ydata[::barb_step_x, ::barb_step_y],
                    u[::barb_step_x, ::barb_step_y],
                    v[::barb_step_x, ::barb_step_y],
                    map_color,
                    **kwargs,
                )
                plt.colorbar(
                    barbs,
                    ax=[ax],
                    label='Wind Speed (' + self._ds[dsname][u_field].attrs['units'] + ')',
                )
            else:
                barbs = ax.barbs(
                    xdata[::barb_step_x, ::barb_step_y],
                    ydata[::barb_step_x, ::barb_step_y],
                    u[::barb_step_x, ::barb_step_y],
                    v[::barb_step_x, ::barb_step_y],
                    **kwargs,
                )

        if day_night_background is True:
            self.day_night_background(subplot_index=subplot_index, dsname=dsname)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )

        ax.set_title(set_title)

        # Set YTitle
        if 'ytitle' in locals():
            ax.set_ylabel(ytitle)

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
            days = self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0]
        else:
            days = (
                self.xrng[subplot_index[0], subplot_index[1], 1]
                - self.xrng[subplot_index[0], subplot_index[1], 0]
            )

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            ax.set_xlabel('Time [UTC]')

        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)
        self.axes[subplot_index] = ax

        return self.axes[subplot_index]

    def plot_time_height_xsection_from_1d_data(
        self,
        data_field,
        pres_field,
        dsname=None,
        subplot_index=(0,),
        set_title=None,
        day_night_background=False,
        num_time_periods=20,
        num_y_levels=20,
        invert_y_axis=True,
        cbar_label=None,
        set_shading='auto',
        **kwargs,
    ):
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
        set_shading : string
            Option to to set the matplotlib.pcolormesh shading parameter.
            Default to 'auto'
        **kwargs : keyword arguments
            Additional keyword arguments will be passed
            into :func:`plt.pcolormesh`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle pointing to the plot.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2'
                'or more datasets in the TimeSeriesDisplay'
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        dim = list(self._ds[dsname][data_field].dims)
        if len(dim) > 1:
            raise ValueError(
                'plot_time_height_xsection_from_1d_data only '
                'supports 1-D datasets. For datasets with 2 or '
                'more dimensions use plot().'
            )

        # Get data and dimensions
        data = self._ds[dsname][data_field].values
        xdata = self._ds[dsname][dim[0]].values

        # What we will do here is do a nearest-neighbor interpolation for each
        # member of the series. Coordinates are time, pressure
        pres = self._ds[dsname][pres_field]
        u_interp = NearestNDInterpolator((xdata, pres.values), data, rescale=True)
        # Mask points where we have no data
        # Count number of unique days
        x_times = pd.date_range(xdata.min(), xdata.max(), periods=num_time_periods)
        y_levels = np.linspace(np.nanmin(pres), np.nanmax(pres), num_y_levels)
        tdata, ydata = np.meshgrid(x_times, y_levels, indexing='ij')
        data = u_interp(tdata, ydata)
        ytitle = ''.join(['(', pres.attrs['units'], ')'])
        units = data_field + ' (' + self._ds[dsname][data_field].attrs['units'] + ')'

        # Get the current plotting axis, add day/night background and plot data
        if self.fig is None:
            self.fig = plt.figure()

        # Set up or get current axes
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        ax = self.axes[subplot_index]

        mesh = ax.pcolormesh(x_times, y_levels, np.transpose(data), shading=set_shading, **kwargs)

        if day_night_background is True:
            self.day_night_background(subplot_index=subplot_index, dsname=dsname)

        # Set Title
        if set_title is None:
            set_title = ' '.join(
                [
                    dsname,
                    'on',
                    dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                ]
            )

        ax.set_title(set_title)

        # Set YTitle
        if 'ytitle' in locals():
            ax.set_ylabel(ytitle)

        # Set X Limit - We want the same time axes for all subplots
        time_rng = [x_times[0], x_times[-1]]

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
            days = self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0]
        else:
            days = (
                self.xrng[subplot_index[0], subplot_index[1], 1]
                - self.xrng[subplot_index[0], subplot_index[1], 0]
            )

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            ax.set_xlabel('Time [UTC]')

        if ydata is not None:
            if cbar_label is None:
                self.add_colorbar(mesh, title=units, subplot_index=subplot_index)
            else:
                self.add_colorbar(mesh, title=cbar_label, subplot_index=subplot_index)
        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)

        return self.axes[subplot_index]

    def time_height_scatter(
        self,
        data_field=None,
        alt_field='alt',
        dsname=None,
        cmap='rainbow',
        alt_label=None,
        cb_label=None,
        subplot_index=(0,),
        plot_alt_field=False,
        cb_friendly=False,
        day_night_background=False,
        set_title=None,
        **kwargs,
    ):
        """
        Create a time series plot of altitude and data variable with
        color also indicating value with a color bar. The Color bar is
        positioned to serve both as the indicator of the color intensity
        and the second y-axis.

        Parameters
        ----------
        data_field : str
            Name of data field in the dataset to plot on second y-axis.
        alt_field : str
            Variable to use for y-axis.
        dsname : str or None
            The name of the datastream to plot.
        cmap : str
            Colorbar color map to use.
        alt_label : str
            Altitude first y-axis label to use. If None, will try to use
            long_name and units.
        cb_label : str
            Colorbar label to use. If not set will try to use
            long_name and units.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        plot_alt_field : boolean
            Set to true to plot the altitude field on the secondary y-axis
        cb_friendly : boolean
            If set to True will use the Homeyer colormap
        day_night_background : boolean
            If set to True will plot the day_night_background
        set_title : str
            Title to set on the plot
        **kwargs : keyword arguments
            Any other keyword arguments that will be passed
            into TimeSeriesDisplay.plot module when the figure
            is made.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Set up or get current plot figure
        if self.fig is None:
            self.fig = plt.figure()

        # Set up or get current axes
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        if cb_friendly:
            cmap = 'HomeyerRainbow'

        ax = self.axes[subplot_index]

        # Get data and dimensions
        data = self._ds[dsname][data_field]
        altitude = self._ds[dsname][alt_field]
        dim = list(self._ds[dsname][data_field].dims)
        xdata = self._ds[dsname][dim[0]]

        if alt_label is None:
            try:
                alt_label = altitude.attrs['long_name'] + ''.join(
                    [' (', altitude.attrs['units'], ')']
                )
            except KeyError:
                alt_label = alt_field

        if cb_label is None:
            try:
                cb_label = data.attrs['long_name'] + ''.join([' (', data.attrs['units'], ')'])
            except KeyError:
                cb_label = data_field

        if 'units' in data.attrs:
            ytitle = ''.join(['(', data.attrs['units'], ')'])
        else:
            ytitle = data_field

        # Set Title
        if set_title is None:
            if isinstance(self._ds[dsname].time.values[0], np.datetime64):
                set_title = ' '.join(
                    [
                        dsname,
                        data_field,
                        'on',
                        dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),
                    ]
                )
            else:
                date_result = search(r'\d{4}-\d{1,2}-\d{1,2}', self._ds[dsname].time.attrs['units'])
                if date_result is not None:
                    set_title = ' '.join([dsname, data_field, 'on', date_result.group(0)])
                else:
                    set_title = ' '.join([dsname, data_field])

        # Plot scatter data
        sc = ax.scatter(xdata.values, data.values, c=data.values, cmap=cmap, **kwargs)

        ax.set_title(set_title)
        if plot_alt_field:
            self.fig.subplots_adjust(left=0.1, right=0.8, bottom=0.15, top=0.925)
            pad = 0.02 + (0.02 * len(str(int(np.nanmax(altitude.values)))))
            cbar = self.fig.colorbar(sc, pad=pad, cmap=cmap)

            ax2 = ax.twinx()
            ax2.set_ylabel(alt_label)
            ax2.scatter(xdata.values, altitude.values, color='black')
        else:
            cbar = self.fig.colorbar(sc, cmap=cmap)

        if day_night_background is True:
            self.day_night_background(subplot_index=subplot_index, dsname=dsname)
        cbar.ax.set_ylabel(cb_label)

        # Set X Limit - We want the same time axes for all subplots
        self.time_rng = [xdata.min().values, xdata.max().values]
        self.set_xrng(self.time_rng, subplot_index)

        # Set X Format
        if len(subplot_index) == 1:
            days = self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0]
        else:
            days = (
                self.xrng[subplot_index[0], subplot_index[1], 1]
                - self.xrng[subplot_index[0], subplot_index[1], 0]
            )
        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel(ytitle)

        self.axes[subplot_index] = ax

        return self.axes[subplot_index]

    def qc_flag_block_plot(
        self,
        data_field=None,
        dsname=None,
        subplot_index=(0,),
        time_rng=None,
        assessment_color=None,
        edgecolor='face',
        set_shading='auto',
        cb_friendly=False,
        **kwargs,
    ):
        """
        Create a time series plot of embedded quality control values
        using broken barh plotting.

        Parameters
        ----------
        data_field : str
            Name of data field in the dataset to plot corresponding quality
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
        assessment_color : dict
            Dictionary lookup to override default assessment to color. Make sure
            assessment work is correctly set with case syntax.
        edgecolor : str or list
            Color name, list of color names or 'face' as defined in matplotlib.axes.Axes.broken_barh
        set_shading : string
            Option to to set the matplotlib.pcolormesh shading parameter.
            Default to 'auto'
        cb_friendly : boolean
            Set to true if you want to use the integrated colorblind friendly
            colors for green/red based on the Homeyer colormap
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.broken_barh`.

        """
        # Color to plot associated with assessment.
        color_lookup = {
            'Bad': 'red',
            'Incorrect': 'red',
            'Indeterminate': 'orange',
            'Suspect': 'orange',
            'Missing': 'darkgray',
            'Not Failing': 'green',
            'Acceptable': 'green',
        }
        if cb_friendly:
            color_lookup['Bad'] = (0.9285714285714286, 0.7130901016453677, 0.7130901016453677)
            color_lookup['Incorrect'] = (0.9285714285714286, 0.7130901016453677, 0.7130901016453677)
            color_lookup['Not Failing'] = (0.0, 0.4240129715562796, 0.4240129715562796)
            color_lookup['Acceptable'] = (0.0, 0.4240129715562796, 0.4240129715562796)
            color_lookup['Indeterminate'] = (1.0, 0.6470588235294118, 0.0)
            color_lookup['Suspect'] = (1.0, 0.6470588235294118, 0.0)
            color_lookup['Missing'] = (0.6627450980392157, 0.6627450980392157, 0.6627450980392157)

        if assessment_color is not None:
            for asses, color in assessment_color.items():
                color_lookup[asses] = color
                if asses == 'Incorrect':
                    color_lookup['Bad'] = color
                if asses == 'Suspect':
                    color_lookup['Indeterminate'] = color

        # Set up list of test names to use for missing values
        missing_val_long_names = [
            'Value equal to missing_value*',
            'Value set to missing_value*',
            'Value is equal to missing_value*',
            'Value is set to missing_value*',
        ]

        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Set up or get current plot figure
        if self.fig is None:
            self.fig = plt.figure()

        # Set up or get current axes
        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        ax = self.axes[subplot_index]

        # Set X Limit - We want the same time axes for all subplots
        data = self._ds[dsname][data_field]
        dim = list(self._ds[dsname][data_field].dims)
        xdata = self._ds[dsname][dim[0]]

        # Get data and attributes
        qc_data_field = self._ds[dsname].qcfilter.check_for_ancillary_qc(
            data_field, add_if_missing=False, cleanup=False
        )
        if qc_data_field is None:
            raise ValueError(f'No quality control ancillary variable in Dataset for {data_field}')

        flag_masks = self._ds[dsname][qc_data_field].attrs['flag_masks']
        flag_meanings = self._ds[dsname][qc_data_field].attrs['flag_meanings']
        flag_assessments = self._ds[dsname][qc_data_field].attrs['flag_assessments']

        # Get time ranges for green blocks
        time_delta = determine_time_delta(xdata.values)
        barh_list_green = reduce_time_ranges(xdata.values, time_delta=time_delta, broken_barh=True)

        # Set background to gray indicating not available data
        ax.set_facecolor('dimgray')

        # Check if plotting 2D data vs 1D data. 2D data will be summarized by
        # assessment category instead of showing each test.
        data_shape = self._ds[dsname][qc_data_field].shape
        if len(data_shape) > 1:
            cur_assessments = list(set(flag_assessments))
            cur_assessments.sort()
            cur_assessments.reverse()
            qc_data = np.full(data_shape, -1, dtype=np.int16)
            plot_colors = []
            tick_names = []

            index = self._ds[dsname][qc_data_field].values == 0
            if index.any():
                qc_data[index] = 0
                plot_colors.append(color_lookup['Not Failing'])
                tick_names.append('Not Failing')

            for ii, assess in enumerate(cur_assessments):
                if assess not in color_lookup:
                    color_lookup[assess] = list(mplcolors.CSS4_COLORS.keys())[ii]
                ii += 1
                assess_data = self._ds[dsname].qcfilter.get_masked_data(
                    data_field, rm_assessments=assess
                )

                if assess_data.mask.any():
                    qc_data[assess_data.mask] = ii
                    plot_colors.append(color_lookup[assess])
                    tick_names.append(assess)

            # Overwrite missing data. Not sure if we want to do this because VAPs set
            # the value to missing but the test is set to Bad. This tries to overcome that
            # by looking for correct test description that would only indicate the values
            # are missing not that they are set to missing by a test... most likely.
            missing_test_nums = []
            for ii, flag_meaning in enumerate(flag_meanings):
                # Check if the bit set is indicating missing data.
                for val in missing_val_long_names:
                    if re_search(val, flag_meaning):
                        test_num = parse_bit(flag_masks[ii])[0]
                        missing_test_nums.append(test_num)
            assess_data = self._ds[dsname].qcfilter.get_masked_data(
                data_field, rm_tests=missing_test_nums
            )
            if assess_data.mask.any():
                qc_data[assess_data.mask] = -1
                plot_colors.append(color_lookup['Missing'])
                tick_names.append('Missing')

            # Create a masked array to allow not plotting where values are missing
            qc_data = np.ma.masked_equal(qc_data, -1)

            dims = self._ds[dsname][qc_data_field].dims
            xvalues = self._ds[dsname][dims[0]].values
            yvalues = self._ds[dsname][dims[1]].values

            cMap = mplcolors.ListedColormap(plot_colors)
            print(plot_colors)
            mesh = ax.pcolormesh(
                xvalues,
                yvalues,
                np.transpose(qc_data),
                cmap=cMap,
                vmin=0,
                shading=set_shading,
            )
            divider = make_axes_locatable(ax)
            # Determine correct placement of words on colorbar
            tick_nums = (
                np.arange(0, len(tick_names) * 2 + 1) / (len(tick_names) * 2) * np.nanmax(qc_data)
            )[1::2]
            cax = divider.append_axes('bottom', size='5%', pad=0.3)
            cbar = self.fig.colorbar(
                mesh,
                cax=cax,
                orientation='horizontal',
                spacing='uniform',
                ticks=tick_nums,
                shrink=0.5,
            )
            cbar.ax.set_xticklabels(tick_names)

            # Set YTitle
            dim_name = list(set(self._ds[dsname][qc_data_field].dims) - {'time'})
            try:
                ytitle = f"{dim_name[0]} ({self._ds[dsname][dim_name[0]].attrs['units']})"
                ax.set_ylabel(ytitle)
            except KeyError:
                pass

            # Add which tests were set as text to the plot
            unique_values = []
            for ii in np.unique(self._ds[dsname][qc_data_field].values):
                unique_values.extend(parse_bit(ii))
            if len(unique_values) > 0:
                unique_values = list(set(unique_values))
                unique_values.sort()
                unique_values = [str(ii) for ii in unique_values]
                self.fig.text(
                    0.5,
                    -0.35,
                    f"QC Tests Tripped: {', '.join(unique_values)}",
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontweight='bold',
                )

        else:
            test_nums = []
            for ii, assess in enumerate(flag_assessments):
                if assess not in color_lookup:
                    color_lookup[assess] = list(mplcolors.CSS4_COLORS.keys())[ii]

                # Plot green data first.
                ax.broken_barh(
                    barh_list_green,
                    (ii, ii + 1),
                    facecolors=color_lookup['Not Failing'],
                    edgecolor=edgecolor,
                    **kwargs,
                )

                # Get test number from flag_mask bitpacked number
                test_nums.append(parse_bit(flag_masks[ii]))
                # Get masked array data to use mask for finding if/where test is set
                data = self._ds[dsname].qcfilter.get_masked_data(data_field, rm_tests=test_nums[-1])
                if np.any(data.mask):
                    # Get time ranges from time and masked data
                    barh_list = reduce_time_ranges(
                        xdata.values[data.mask], time_delta=time_delta, broken_barh=True
                    )
                    # Check if the bit set is indicating missing data. If so change
                    # to different plotting color than what is in flag_assessments.
                    for val in missing_val_long_names:
                        if re_search(val, flag_meanings[ii]):
                            assess = 'Missing'
                            break
                    # Lay down blocks of tripped tests using correct color
                    ax.broken_barh(
                        barh_list,
                        (ii, ii + 1),
                        facecolors=color_lookup[assess],
                        edgecolor=edgecolor,
                        **kwargs,
                    )

                # Add test description to plot.
                ax.text(xdata.values[0], ii + 0.5, ' ' + flag_meanings[ii], va='center')

            # Change y ticks to test number
            plt.yticks(
                [ii + 0.5 for ii in range(0, len(test_nums))],
                labels=['Test ' + str(ii[0]) for ii in test_nums],
            )
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
                days = self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0]
            else:
                days = (
                    self.xrng[subplot_index[0], subplot_index[1], 1]
                    - self.xrng[subplot_index[0], subplot_index[1], 0]
                )

            myFmt = common.get_date_format(days)
            ax.xaxis.set_major_formatter(myFmt)
            self.time_fmt = myFmt

        return self.axes[subplot_index]

    def fill_between(
        self,
        field,
        dsname=None,
        subplot_index=(0,),
        set_title=None,
        **kwargs,
    ):
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
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.plot` (1D timeseries) or
            :func:`plt.pcolormesh` (2D timeseries).

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Get data and dimensions
        data = self._ds[dsname][field]
        dim = list(self._ds[dsname][field].dims)
        xdata = self._ds[dsname][dim[0]]

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
        ax = self.axes[subplot_index]

        ax.fill_between(xdata.values, data, **kwargs)

        # Set X Format
        if len(subplot_index) == 1:
            days = self.xrng[subplot_index, 1] - self.xrng[subplot_index, 0]
        else:
            days = (
                self.xrng[subplot_index[0], subplot_index[1], 1]
                - self.xrng[subplot_index[0], subplot_index[1], 0]
            )

        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)

        # Set X format - We want the same time axes for all subplots
        if not hasattr(self, 'time_fmt'):
            self.time_fmt = myFmt

        # Put on an xlabel, but only if we are making the bottom-most plot
        if subplot_index[0] == self.axes.shape[0] - 1:
            ax.set_xlabel('Time [UTC]')

        # Set YTitle
        ax.set_ylabel(ytitle)

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
        ax.set_title(set_title)
        self.axes[subplot_index] = ax
        return self.axes[subplot_index]
