"""
Stores the class for XSectionDisplay.

"""

# Import third party libraries
import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Import Local Libs
from ..utils import data_utils
from .plot import Display


class XSectionDisplay(Display):
    """
    Plots cross sections of multidimensional datasets. The data
    must be able to be sliced into a 2 dimensional slice using the
    xarray :func:`xarray.Dataset.sel` and :func:`xarray.Dataset.isel` commands.

    This is inherited from the :func:`act.plotting.Display`
    class and has therefore has the same attributes as that class.
    See :func:`act.plotting.Display`
    for more information. There are no additional attributes or parameters
    to this class.

    In order to create geographic plots, ACT needs the Cartopy package to be
    installed on your system. More information about
    Cartopy go here:https://scitools.org.uk/cartopy/docs/latest/.

    Examples
    --------
    For example, if you only want to do a cross section through the first
    time period of a 3D dataset called :code:`ir_temperature`, you would
    do the following in xarray:

    .. code-block:: python

        time_slice = my_ds['ir_temperature'].isel(time=0)

    The methods of this class support passing in keyword arguments into
    xarray :func:`xarray.Dataset.sel` and :func:`xarray.Dataset.isel` commands
    so that new datasets do not need to be created when slicing by specific time
    periods or spatial slices. For example, to plot the first time period
    from :code:`my_ds`, simply do:

    .. code-block:: python

        xsection = XSectionDisplay(my_ds, figsize=(15, 8))
        xsection.plot_xsection_map(
            None, 'ir_temperature', vmin=220, vmax=300,
            cmap='Greys', x='longitude', y='latitude',
            isel_kwargs={'time': 0})

    Here, the array is sliced by the first time period as specified
    in :code:`isel_kwargs`. The other keyword arguments are standard keyword
    arguments taken by :func:`matplotlib.pyplot.pcolormesh`.

    """
    def __init__(self, obj, subplot_shape=(1,),
                 ds_name=None, **kwargs):
        super().__init__(obj, None, ds_name, **kwargs)
        self.add_subplots(subplot_shape)

    def set_subplot_to_map(self, subplot_index):
        total_num_plots = self.axes.shape

        if len(total_num_plots) == 2:
            second_number = total_num_plots[0]
            j = subplot_index[1]
        else:
            second_number = 1
            j = 0

        third_number = second_number * subplot_index[0] + j + 1

        self.axes[subplot_index] = plt.subplot(
            total_num_plots[0], second_number, third_number,
            projection=ccrs.PlateCarree())

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
                                 dtype=xrng[0].dtype)
        elif not hasattr(self, 'xrng') and len(self.axes.shape) == 1:
            self.xrng = np.zeros((self.axes.shape[0], 2),
                                 dtype=xrng[0].dtype)

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
            raise RuntimeError("set_yrng requires the plot to be displayed.")

        if not hasattr(self, 'yrng') and len(self.axes.shape) == 2:
            self.yrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2),
                                 dtype=yrng[0].dtype)
        elif not hasattr(self, 'yrng') and len(self.axes.shape) == 1:
            self.yrng = np.zeros((self.axes.shape[0], 2), dtype=yrng[0].dtype)

        if yrng[0] == yrng[1]:
            yrng[1] = yrng[1] + 1

        self.axes[subplot_index].set_ylim(yrng)

        self.yrng[subplot_index, :] = yrng

    def plot_xsection(self, dsname, varname, x=None, y=None,
                      subplot_index=(0, ),
                      sel_kwargs=None, isel_kwargs=None,
                      **kwargs):
        """
        This function plots a cross section whose x and y coordinates are
        specified by the variable names either provided by the user or
        automatically detected by xarray.

        Parameters
        ----------
        dsname : str or None
            The name of the datastream to plot from. Set to None to have
            ACT attempt to automatically detect this.
        varname : str
            The name of the variable to plot.
        x : str or None
            The name of the x coordinate variable.
        y : str or None
            The name of the y coordinate variable.
        subplot_index : tuple
            The index of the subplot to create the plot in.
        sel_kwargs : dict
            The keyword arguments to pass into :py:func:`xarray.DataArray.sel`
            This is useful when your data is in 3 or more dimensions and you
            want to only view a cross section on a specific x-y plane. For more
            information on how to use xarray's .sel and .isel functionality
            to slice datasets, see the documentation on :func:`xarray.DataArray.sel`.
        isel_kwargs : dict
            The keyword arguments to pass into :py:func:`xarray.DataArray.sel`
        **kwargs : keyword arguments
            Additional keyword arguments will be passed into
            :func:`xarray.DataArray.plot`.

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.

        """
        if dsname is None and len(self._obj.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._obj.keys())[0]
        temp_ds = self._obj[dsname].copy()

        if sel_kwargs is not None:
            temp_ds = temp_ds.sel(**sel_kwargs, method='nearest')

        if isel_kwargs is not None:
            temp_ds = temp_ds.isel(**isel_kwargs)

        if((x is not None and y is None) or (y is None and x is not None)):
            raise RuntimeError("Both x and y must be specified if we are" +
                               "not trying to automatically detect them!")

        if x is not None:
            coord_list = {}
            x_coord_dim = temp_ds[x].dims[0]
            coord_list[x] = x_coord_dim
            y_coord_dim = temp_ds[y].dims[0]
            coord_list[y] = y_coord_dim
            new_ds = data_utils.assign_coordinates(
                temp_ds, coord_list)
            my_dataarray = new_ds[varname]
        else:
            my_dataarray = temp_ds[varname]

        coord_keys = [key for key in my_dataarray.coords.keys()]
        # X-array will sometimes shorten latitude and longitude variables
        if x == 'longitude' and x not in coord_keys:
            xc = 'lon'
        else:
            xc = x
        if y == 'latitude' and y not in coord_keys:
            yc = 'lat'
        else:
            yc = y

        if x is None:
            ax = my_dataarray.plot(ax=self.axes[subplot_index], **kwargs)
        else:
            ax = my_dataarray.plot(ax=self.axes[subplot_index], x=xc, y=yc, **kwargs)

        the_coords = [the_keys for the_keys in my_dataarray.coords.keys()]
        if x is None:
            x = the_coords[0]
        else:
            x = coord_list[x]

        if y is None:
            y = the_coords[1]
        else:
            y = coord_list[y]

        xrng = self.axes[subplot_index].get_xlim()
        self.set_xrng(xrng, subplot_index)
        yrng = self.axes[subplot_index].get_ylim()
        self.set_yrng(yrng, subplot_index)
        del temp_ds
        return ax

    def plot_xsection_map(self, dsname, varname,
                          subplot_index=(0, ), coastlines=True,
                          background=False, **kwargs):
        """
        Plots a cross section of 2D data on a geographical map.

        Parameters
        ----------
        dsname : str or None
            The name of the datastream to plot from. Set to None
            to have ACT attempt to automatically detect this.
        varname : str
            The name of the variable to plot.
        subplot_index : tuple
            The index of the subplot to plot inside.
        coastlines : bool
            Set to True to plot the coastlines.
        background : bool
            Set to True to plot a stock image background.
        **kwargs : keyword arguments
            Additional keyword arguments will be passed into
            :func:`act.plotting.XSectionDisplay.plot_xsection`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.

        """
        if not CARTOPY_AVAILABLE:
            raise ImportError("Cartopy needs to be installed in order to plot " +
                              "cross sections on maps!")

        self.set_subplot_to_map(subplot_index)
        self.plot_xsection(dsname, varname, subplot_index=subplot_index, **kwargs)
        xlims = self.xrng[subplot_index].flatten()
        ylims = self.yrng[subplot_index].flatten()
        self.axes[subplot_index].set_xticks(
            np.linspace(round(xlims[0], 0), round(xlims[1], 0), 10))
        self.axes[subplot_index].set_yticks(
            np.linspace(round(ylims[0], 0), round(ylims[1], 0), 10))

        if coastlines:
            self.axes[subplot_index].coastlines(resolution='10m')

        if background:
            self.axes[subplot_index].stock_img()

        return self.axes[subplot_index]
