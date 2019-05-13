# Import third party libraries
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# Import Local Libs
from ..utils import data_utils
from .plot import Display


class XSectionDisplay(Display):
    """
    Plots cross sections of multidimensional datasets.

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
        self.xrng[subplot_index, :] = np.array(xrng)

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

    def plot_xsection(self, dsname, varname, x=None, y=None,
                      subplot_index=(0, ),
                      sel_kwargs=None, isel_kwargs=None,
                      **kwargs):
        """
        This function plots a cross section whose x and y
        coordinates are specified by the variable
        names either provided by the user or automatically
        detected by xarray.

        Parameters
        ----------
        dsname: str or None
            The name of the datastream to plot from. Set to
            None to have ACT attempt to automatically
            detect this.
        varname: str
            The name of the variable to plot
        x: str or None
            The name of the x coordinate variable
        y: str or None
            The name of the y coordinate variable
        subplot_index: tuple
            The index of the subplot to create the plot in
        sel_kwargs: dict
            The keyword arguments to pass into :py:func:`xarray.DataArray.sel`
            This is useful when your data is in 3 or more dimensions and you
            want to only view a cross section on a specific x-y plane. For more
            information on how to use xarray's .sel and .isel functionality
            to slice datasets, see the documentation on :func:`xarray.DataArray.sel`.
        isel_kwargs:
            The keyword arguments to pass into :py:func:`xarray.DataArray.sel`
        kwargs:
            Additional keyword arguments will be passed into
            :func:`xarray.DataArray.plot`

        Returns
        =======
        ax: matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.

        """
        if dsname is None and len(self._arm.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._arm.keys())[0]
        temp_ds = self._arm[dsname].copy()

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

        ax = my_dataarray.plot(ax=self.axes[subplot_index], **kwargs)
        the_coords = [the_keys for the_keys in my_dataarray.coords.keys()]
        if x is None:
            x = the_coords[0]
        else:
            x = coord_list[x]

        if y is None:
            y = the_coords[1]
        else:
            y = coord_list[y]

        xrng = [np.nanmin(my_dataarray[x].values),
                np.nanmax(my_dataarray[x].values)]
        self.set_xrng(xrng, subplot_index)

        yrng = [np.nanmin(my_dataarray[y].values),
                np.nanmax(my_dataarray[y].values)]
        self.set_yrng(yrng, subplot_index)
        del temp_ds
        return ax

    def plot_xsection_map(self, dsname, varname,
                          subplot_index=(0, ), coastlines=True,
                          background=False, **kwargs):
        """
        Plots a cross section of 2D data on a geographical map

        Parameters
        ==========
        dsname: str or None
            The name of the datastream to plot from. Set to None
            to have ACT attempt to automatically detect this.
        varname: str
            The name of the variable to plot.
        subplot_index: tuple
            The index of the subplot to plot inside.
        coastlines: bool
            Set to True to plot the coastlines.
        background: bool
            Set to True to plot a stock image background
        kwargs:
            Additional keyword arguments will be passed into
            :func:`act.plotting.XSectionDisplay.plot_xsection`

        Returns
        =======
        ax: matplotlib axis handle
            The matplotlib axis handle corresponding to the plot.
        """
        self.set_subplot_to_map(subplot_index)
        self.plot_xsection(dsname, varname, subplot_index=subplot_index, **kwargs)
        xlims = self.xrng[subplot_index].flatten()
        ylims = self.yrng[subplot_index].flatten()
        self.axes[subplot_index].set_xticks(np.linspace(round(xlims[0], 0), round(xlims[1], 0), 10))
        self.axes[subplot_index].set_yticks(np.linspace(round(ylims[0], 0), round(ylims[1], 0), 10))

        if coastlines:
            self.axes[subplot_index].coastlines(resolution='10m')

        if background:
            self.axes[subplot_index].stock_img()

        return self.axes[subplot_index]
