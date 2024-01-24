"""
Stores the class for GeographicPlotDisplay.

"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot import Display

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import img_tiles

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


class GeographicPlotDisplay(Display):
    """
    A class for making geographic tracer plot of aircraft, ship or other moving
    platform plot.

    This is inherited from the :func:`act.plotting.Display`
    class and has therefore has the same attributes as that class.
    See :func:`act.plotting.Display`
    for more information. There are no additional attributes or parameters
    to this class.

    In order to create geographic plots, ACT needs the Cartopy package to be
    installed on your system. More information about
    Cartopy go here:https://scitools.org.uk/cartopy/docs/latest/ .

    """

    def __init__(self, ds, ds_name=None, **kwargs):
        if not CARTOPY_AVAILABLE:
            raise ImportError(
                'Cartopy needs to be installed on your ' 'system to make geographic display plots.'
            )
        super().__init__(ds, ds_name, **kwargs)
        if self.fig is None:
            self.fig = plt.figure(**kwargs)

    def geoplot(
        self,
        data_field=None,
        lat_field='lat',
        lon_field='lon',
        dsname=None,
        cbar_label=None,
        title=None,
        projection=None,
        plot_buffer=0.08,
        img_tile=None,
        img_tile_args={},
        tile=8,
        cartopy_feature=None,
        cmap='rainbow',
        text=None,
        gridlines=True,
        **kwargs,
    ):
        """
        Creates a latitude and longitude plot of a time series data set with
        data values indicated by color and described with a colorbar.
        Latitude values must be in degree north (-90 to 90) and
        longitude must be in degree east (-180 to 180).

        Parameters
        ----------
        data_field : str
            Name of data field in the dataset to plot.
        lat_field : str
            Name of latitude field in the dataset to use.
        lon_field : str
            Name of longitude field in the dataset to use.
        dsname : str or None
            The name of the datastream to plot. Set to None to make ACT
            attempt to automatically determine this.
        cbar_label : str
            Label to use with colorbar. If set to None will attempt
            to create label from long_name and units.
        title : str
            Plot title.
        projection : cartopy.crs object
            Project to use on plot. See
            https://scitools.org.uk/cartopy/docs/latest/reference/projections.html?highlight=projections
        plot_buffer : float
            Buffer to add around data on plot in lat and lon dimension.
        img_tile : str
            Image to use for the plot background. Set to None to not use
            background image. For all image background types, see:
            https://scitools.org.uk/cartopy/docs/v0.16/cartopy/io/img_tiles.html
            Default is None.
        img_tile_args : dict
            Keyword arguments for the chosen img_tile. These arguments can be
            found for the corresponding img_tile here:
            https://scitools.org.uk/cartopy/docs/v0.16/cartopy/io/img_tiles.html
            Default is an empty dictionary.
        tile : int
            Tile zoom to use with background image. Higher number indicates
            more resolution. A value of 8 is typical for a normal sonde plot.
        cartopy_feature : list of str or str
            Cartopy feature to add to plot.
        cmap : str
            Color map to use for colorbar.
        text : dictionary
            Dictionary of {text:[lon,lat]} to add to plot. Can have more
            than one set of text to add.
        gridlines : boolean
            Use latitude and longitude gridlines.
        **kwargs : keyword arguments
            Any other keyword arguments that will be passed
            into :func:`matplotlib.pyplot.scatter` when the figure
            is made. See the matplotlib documentation for further details
            on what keyword arguments are available.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the GeographicPlotDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        if data_field is None:
            raise ValueError('You must enter the name of the data ' 'to be plotted.')

        if projection is None:
            if CARTOPY_AVAILABLE:
                projection = ccrs.PlateCarree()

        # Extract data from the dataset
        try:
            lat = self._ds[dsname][lat_field].values
        except KeyError:
            raise ValueError(
                (
                    'You will need to provide the name of the '
                    "field if not '{}' to use for latitude "
                    'data.'
                ).format(lat_field)
            )
        try:
            lon = self._ds[dsname][lon_field].values
        except KeyError:
            raise ValueError(
                (
                    'You will need to provide the name of the '
                    "field if not '{}' to use for longitude "
                    'data.'
                ).format(lon_field)
            )

        # Set up metadata information for display on plot
        if cbar_label is None:
            try:
                cbar_label = (
                    self._ds[dsname][data_field].attrs['long_name']
                    + ' ('
                    + self._ds[dsname][data_field].attrs['units']
                    + ')'
                )
            except KeyError:
                cbar_label = data_field

        lat_limits = [np.nanmin(lat), np.nanmax(lat)]
        lon_limits = [np.nanmin(lon), np.nanmax(lon)]
        box_size = np.max([np.abs(np.diff(lat_limits)), np.abs(np.diff(lon_limits))])
        bx_buf = box_size * plot_buffer

        lat_center = np.sum(lat_limits) / 2.0
        lon_center = np.sum(lon_limits) / 2.0

        lat_limits = [
            lat_center - box_size / 2.0 - bx_buf,
            lat_center + box_size / 2.0 + bx_buf,
        ]
        lon_limits = [
            lon_center - box_size / 2.0 - bx_buf,
            lon_center + box_size / 2.0 + bx_buf,
        ]

        data = self._ds[dsname][data_field].values

        # Create base plot projection
        ax = plt.axes(projection=projection)
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.93)
        ax.set_extent([lon_limits[0], lon_limits[1], lat_limits[0], lat_limits[1]], crs=projection)

        if title is None:
            try:
                dim = list(self._ds[dsname][data_field].dims)
                ts = pd.to_datetime(str(self._ds[dsname][dim[0]].values[0]))
                date = ts.strftime('%Y-%m-%d')
                time_str = ts.strftime('%H:%M:%S')
                plt.title(' '.join([dsname, 'at', date, time_str]))
            except NameError:
                plt.title(dsname)
        else:
            plt.title(title)

        if img_tile is not None:
            tiler = getattr(img_tiles, img_tile)(**img_tile_args)
            ax.add_image(tiler, tile)

        colorbar_map = None
        if cmap is not None:
            colorbar_map = matplotlib.colormaps.get_cmap(cmap)
        sc = ax.scatter(lon, lat, c=data, cmap=colorbar_map, **kwargs)
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(cbar_label)
        if cartopy_feature is not None:
            if isinstance(cartopy_feature, str):
                cartopy_feature = [cartopy_feature]
            cartopy_feature = [ii.upper() for ii in cartopy_feature]
            if 'STATES' in cartopy_feature:
                ax.add_feature(cfeature.STATES.with_scale('10m'))
            if 'LAND' in cartopy_feature:
                ax.add_feature(cfeature.LAND)
            if 'OCEAN' in cartopy_feature:
                ax.add_feature(cfeature.OCEAN)
            if 'COASTLINE' in cartopy_feature:
                ax.add_feature(cfeature.COASTLINE)
            if 'BORDERS' in cartopy_feature:
                ax.add_feature(cfeature.BORDERS, linestyle=':')
            if 'LAKES' in cartopy_feature:
                ax.add_feature(cfeature.LAKES, alpha=0.5)
            if 'RIVERS' in cartopy_feature:
                ax.add_feature(cfeature.RIVERS)
        if text is not None:
            for label, location in text.items():
                ax.plot(location[0], location[1], marker='*', color='black')
                ax.text(location[0], location[1], label, color='black')

        if gridlines:
            if projection == ccrs.PlateCarree() or projection == ccrs.Mercator:
                gl = ax.gridlines(
                    crs=projection,
                    draw_labels=True,
                    linewidth=1,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                )
                gl.top_labels = False
                gl.left_labels = True
                gl.bottom_labels = True
                gl.right_labels = False
                gl.xlabel_style = {'size': 6, 'color': 'gray'}
                gl.ylabel_style = {'size': 6, 'color': 'gray'}
            else:
                # Labels are only currently supported for PlateCarree and Mercator
                gl = ax.gridlines(
                    draw_labels=False,
                    linewidth=1,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                )

        return ax
