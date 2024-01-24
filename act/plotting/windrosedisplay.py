"""
Stores the class for WindRoseDisplay.

"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Import Local Libs
from ..utils import datetime_utils as dt_utils
from .plot import Display


class WindRoseDisplay(Display):
    """
    A class for handing wind rose plots.

    This is inherited from the :func:`act.plotting.Display`
    class and has therefore has the same attributes as that class.
    See :func:`act.plotting.Display`
    for more information. There are no additional attributes or parameters
    to this class.

    Examples
    --------
    To create a WindRoseDisplay object, simply do:

    .. code-block :: python

        sonde_ds = act.io.arm.read_arm_netcdf('sonde_data.nc')
        WindDisplay = act.plotting.WindRoseDisplay(sonde_ds, figsize=(8,10))

    """

    def __init__(self, ds, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(ds, subplot_shape, ds_name, subplot_kw=dict(projection='polar'), **kwargs)

    def set_thetarng(self, trng=(0.0, 360.0), subplot_index=(0,)):
        """
        Sets the theta range of the wind rose plot.

        Parameters
        ----------
        trng : 2-tuple
            The range (in degrees).
        subplot_index : 2-tuple
            The index of the subplot to set the degree range of.

        """
        if self.axes is not None:
            self.axes[subplot_index].set_thetamin(trng[0])
            self.axes[subplot_index].set_thetamax(trng[1])
            self.trng = trng
        else:
            raise RuntimeError('Axes must be initialized before' + ' changing limits!')

    def set_rrng(self, rrng, subplot_index=(0,)):
        """
        Sets the range of the radius of the wind rose plot.

        Parameters
        ----------
        rrng : 2-tuple
            The range for the plot radius (in %).
        subplot_index : 2-tuple
            The index of the subplot to set the radius range of.

        """
        if self.axes is not None:
            self.axes[subplot_index].set_rmin(rrng[0])
            self.axes[subplot_index].set_rmax(rrng[1])
            self.rrng = rrng
        else:
            raise RuntimeError('Axes must be initialized before' + ' changing limits!')

    def plot(
        self,
        dir_field,
        spd_field,
        dsname=None,
        subplot_index=(0,),
        cmap=None,
        set_title=None,
        num_dirs=20,
        spd_bins=None,
        tick_interval=3,
        legend_loc=0,
        legend_bbox=None,
        legend_title=None,
        calm_threshold=1.0,
        **kwargs,
    ):
        """
        Makes the wind rose plot from the given dataset.

        Parameters
        ----------
        dir_field : str
            The name of the field representing the wind direction (in degrees).
        spd_field : str
            The name of the field representing the wind speed.
        dsname : str
            The name of the datastream to plot from. Set to None to
            let ACT automatically try to determine this.
        subplot_index : 2-tuple
            The index of the subplot to place the plot on.
        cmap : str or matplotlib colormap
            The name of the matplotlib colormap to use.
        set_title : str
            The title of the plot.
        num_dirs : int
            The number of directions to split the wind rose into.
        spd_bins : 1D array-like
            The bin boundaries to sort the wind speeds into.
        tick_interval : int
            The interval (in %) for the ticks on the radial axis.
        legend_loc : int
            Legend location using matplotlib legend code
        legend_bbox : tuple
            Legend bounding box coordinates
        legend_title : string
            Legend title
        calm_threshold : float
            Winds below this threshold are considered to be calm.
        **kwargs : keyword arguments
            Additional keyword arguments will be passed into :func:plt.bar

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.

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
        dir_data = self._ds[dsname][dir_field].values
        spd_data = self._ds[dsname][spd_field].values

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

        deg_width = 360.0 / num_dirs
        dir_bins_mid = np.linspace(0.0, 360.0 - 3 * deg_width / 2.0, num_dirs)
        wind_hist = np.zeros((num_dirs, len(spd_bins) - 1))

        for i in range(num_dirs):
            if i == 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'invalid value encountered in.*')
                    the_range = np.logical_or(
                        dir_data < deg_width / 2.0, dir_data > 360.0 - deg_width / 2.0
                    )
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'invalid value encountered in.*')
                    the_range = np.logical_and(
                        dir_data >= dir_bins_mid[i] - deg_width / 2,
                        dir_data <= dir_bins_mid[i] + deg_width / 2,
                    )
            hist, bins = np.histogram(spd_data[the_range], spd_bins)
            wind_hist[i] = hist

        wind_hist = wind_hist / np.sum(wind_hist) * 100
        mins = np.deg2rad(dir_bins_mid)

        # Do the first level
        if 'units' in self._ds[dsname][spd_field].attrs.keys():
            units = self._ds[dsname][spd_field].attrs['units']
        else:
            units = ''
        the_label = '%3.1f' % spd_bins[0] + '-' + '%3.1f' % spd_bins[1] + ' ' + units
        our_cmap = matplotlib.colormaps.get_cmap(cmap)
        our_colors = our_cmap(np.linspace(0, 1, len(spd_bins)))

        ax = self.axes[subplot_index]

        bars = [
            ax.bar(
                mins,
                wind_hist[:, 0],
                bottom=0,
                label=the_label,
                width=0.8 * np.deg2rad(deg_width),
                color=our_colors[0],
                **kwargs,
            )
        ]
        for i in range(1, len(spd_bins) - 1):
            the_label = '%3.1f' % spd_bins[i] + '-' + '%3.1f' % spd_bins[i + 1] + ' ' + units
            # Changing the bottom to be a sum of the previous speeds so that
            # it positions it correctly - Adam Theisen
            bars.append(
                ax.bar(
                    mins,
                    wind_hist[:, i],
                    label=the_label,
                    bottom=np.sum(wind_hist[:, :i], axis=1),
                    width=0.8 * np.deg2rad(deg_width),
                    color=our_colors[i],
                    **kwargs,
                )
            )
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, title=legend_title)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # Add an annulus with text stating % of time calm
        pct_calm = np.sum(spd_data <= calm_threshold) / len(spd_data) * 100
        ax.set_rorigin(-2.5)
        ax.annotate('%3.2f%%\n calm' % pct_calm, xy=(0, -2.5), ha='center', va='center')

        # Set the ticks to be nice numbers
        tick_max = tick_interval * round(np.nanmax(np.cumsum(wind_hist, axis=1)) / tick_interval)
        rticks = np.arange(0, tick_max, tick_interval)
        rticklabels = [('%d' % x + '%') for x in rticks]
        ax.set_rticks(rticks)
        ax.set_yticklabels(rticklabels)

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

        self.axes[subplot_index] = ax

        return ax

    def plot_data(
        self,
        dir_field,
        spd_field,
        data_field,
        dsname=None,
        subplot_index=(0,),
        plot_type='Line',
        line_color=None,
        set_title=None,
        num_dirs=30,
        num_data_bins=30,
        calm_threshold=1.0,
        line_plot_calc='mean',
        clevels=30,
        contour_type='count',
        cmap=None,
        **kwargs,
    ):
        """
        Makes a data rose plot in line or boxplot form from the given data.

        Parameters
        ----------
        dir_field : str
            The name of the field representing the wind direction (in degrees).
        spd_field : str
            The name of the field representing the wind speed.
        data_field : str
            Name of the field to plot.  Default is to plot mean values.
        dsname : str
            The name of the datastream to plot from. Set to None to
            let ACT automatically try to determine this.
        subplot_index : 2-tuple
            The index of the subplot to place the plot on.
        plot_type : str
            Type of plot to create.  Defaults to a line plot but the full options include
            'line', 'contour', and 'boxplot'
        line_color : str
            Color to use for the line
        set_title : str
            The title of the plot.
        num_dirs : int
            The number of directions to split the wind rose into.
        num_data_bins : int
            The number of bins to use for data processing if doing a contour plot
        calm_threshold : float
            Winds below this threshold are considered to be calm.
        line_plot_calc : str
            What values to display for the line plot.  Defaults to 'mean',
            but other options are 'median' and 'stdev'
        clevels : int
            Number of contour levels to plot
        contour_type : str
            Type of contour plot to do.  Default is 'count' which displays a
            heatmap of where values are occuring most along with wind directions
            The other option is 'mean' which will do a wind direction x wind speed
            plot with the contours of the mean values for each wind dir/speed.
            num_data_bins will be used for number of wind speed bins
        cmap : str or matplotlib colormap
            The name of the matplotlib colormap to use.
        **kwargs : keyword arguments
            Additional keyword arguments will be passed into :func:plt.bar

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.

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
        # Throw out calm winds for the analysis
        ds = self._ds[dsname]
        ds = ds.where(ds[spd_field] >= calm_threshold)
        dir_data = ds[dir_field].values
        data = ds[data_field].values

        # Set the bins
        dir_bins_mid = np.linspace(0.0, 360.0, num_dirs + 1)

        # Run through the data and bin based on the wind direction and plot type
        arr = []
        bins = []
        for i, d in enumerate(dir_bins_mid):
            if i < len(dir_bins_mid) - 1:
                idx = np.where((dir_data > d) & (dir_data <= dir_bins_mid[i + 1]))[0]
                bins.append(d + (dir_bins_mid[i + 1] - d) / 2.0)
            else:
                idx = np.where((dir_data > d) & (dir_data <= 360.0))[0]
                bins.append(d + (360.0 - d) / 2.0)

            if plot_type == 'line':
                if line_plot_calc == 'mean':
                    arr.append(np.nanmean(data[idx]))
                    plot_type_str = 'Mean of'
                elif line_plot_calc == 'median':
                    arr.append(np.nanmedian(data[idx]))
                    plot_type_str = 'Median of'
                elif line_plot_calc == 'stdev':
                    plot_type_str = 'Standard Deviation of'
                    arr.append(np.nanstd(data[idx]))
                else:
                    raise ValueError('Please pick an available option')
            elif plot_type == 'boxplot':
                arr.append(data[idx])

        # Plot data for each plot type
        if plot_type == 'line':
            # Add the first values to the end of the array to have a
            # complete circle
            bins.append(bins[0])
            arr.append(arr[0])
            self.axes[subplot_index].plot(np.deg2rad(bins), arr, **kwargs)
        elif plot_type == 'boxplot':
            # Plot boxplot
            self.axes[subplot_index].boxplot(
                arr, positions=np.deg2rad(bins), showmeans=False, **kwargs
            )
            if bins[-1] == 360:
                bins[-1] = 0
            self.axes[subplot_index].xaxis.set_ticklabels(np.ceil(bins))
            plot_type_str = 'Boxplot of'
        elif plot_type == 'contour':
            # Calculate a histogram to plot out a contour for
            if contour_type == 'count':
                idx = np.where((~np.isnan(dir_data)) & (~np.isnan(data)))[0]
                hist, xedges, yedges = np.histogram2d(
                    dir_data[idx], data[idx], bins=[num_dirs, num_data_bins]
                )
                hist = np.insert(hist, -1, hist[0], axis=0)
                cplot = self.axes[subplot_index].contourf(
                    np.deg2rad(xedges),
                    yedges[0:-1],
                    np.transpose(hist),
                    cmap=cmap,
                    levels=clevels,
                    **kwargs,
                )
                plot_type_str = 'Heatmap of'
                cbar = self.fig.colorbar(cplot, ax=self.axes[subplot_index])
                cbar.ax.set_ylabel('Count')
            elif contour_type == 'mean':
                # Produce direction (x-axis) and speed (y-axis) plots displaying the mean
                # as the contours.
                spd_data = ds[spd_field].values
                spd_bins = np.linspace(0, ds[spd_field].max(), num_data_bins + 1)
                spd_bins = np.insert(spd_bins, 1, calm_threshold)
                #  Set up an array and cycle through the data, binning them by speed/direction
                mean_data = np.zeros([len(bins), len(spd_bins)])
                for i in range(len(bins) - 1):
                    for j in range(len(spd_bins)):
                        if j < len(spd_bins) - 1:
                            idx = np.where(
                                (spd_data >= spd_bins[j])
                                & (spd_data < spd_bins[j + 1])
                                & (dir_data >= bins[i])
                                & (dir_data < bins[i + 1])
                            )[0]
                        else:
                            idx = np.where(
                                (spd_data >= spd_bins[j])
                                & (dir_data >= bins[i])
                                & (dir_data < bins[i + 1])
                            )[0]
                        mean_data[i, j] = np.nanmean(data[idx])

                # Necessary to produce the full polar contour without having gaps
                mean_data = np.insert(mean_data, -1, mean_data[0, :], axis=0)
                bins.append(bins[0])
                mean_data[-1, :] = mean_data[0, :]

                # In order to properly handle vmin/vmax in contours, need to adjust
                # the levels plotted and remove the keywords to contourf
                vmin = np.nanmin(mean_data)
                vmax = np.nanmax(mean_data)
                if 'vmin' in kwargs:
                    vmin = kwargs.get('vmin')
                    kwargs.pop('vmin', None)
                if 'vmax' in kwargs:
                    vmax = kwargs.get('vmax')
                    kwargs.pop('vmax', None)

                clevels = np.linspace(vmin, vmax, clevels)
                cplot = self.axes[subplot_index].contourf(
                    np.deg2rad(bins),
                    spd_bins,
                    np.transpose(mean_data),
                    cmap=cmap,
                    levels=clevels,
                    extend='both',
                    **kwargs,
                )
                plot_type_str = 'Mean of'
                cbar = self.fig.colorbar(cplot, ax=self.axes[subplot_index])
                cbar.ax.set_ylabel('Mean')
        else:
            raise ValueError('Please choose an available plot type')

        # Set axis parameters so that it's a standard wind rose style
        self.axes[subplot_index].set_theta_zero_location('N')
        self.axes[subplot_index].set_theta_direction(-1)

        # Set Title
        sdate = (dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),)
        edate = (dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[-1]),)

        if sdate == edate:
            date_str = 'on ' + sdate[0]
        else:
            date_str = 'from ' + sdate[0] + ' to ' + edate[0]
        if 'units' in ds[data_field].attrs:
            units = ds[data_field].attrs['units']
        else:
            units = ''
        if set_title is None:
            set_title = ' '.join(
                [plot_type_str, data_field + ' (' + units + ')', 'by\n', dir_field, date_str]
            )
        self.axes[subplot_index].set_title(set_title)
        plt.tight_layout(h_pad=1.05)

        return self.axes[subplot_index]
