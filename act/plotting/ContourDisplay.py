"""
act.plotting.ContourDisplay
------------------------------

Stores the class for ContourDisplay.

"""

from scipy.interpolate import Rbf
import numpy as np

# Import Local Libs
from .plot import Display


class ContourDisplay(Display):
    """
    This subclass contains routines that are specific to plotting
    contour plots from data. It is inherited from Display and therefore
    contains all of Display's attributes and methods.

    """
    def __init__(self, obj, subplot_shape=(1,), ds_name=None, **kwargs):
        super().__init__(obj, subplot_shape, ds_name, **kwargs)

    def create_contour(self, fields=None, time=None, function='cubic',
                       subplot_index=(0,), grid_delta=(0.01, 0.01),
                       grid_buffer=0.1, **kwargs):
        """
        Extracts, grids, and creates a contour plot. If subplots have not been
        added yet, an axis will be created assuming that there is only going
        to be one plot.

        Parameters
        ----------
        fields : dict
            Dictionary of fields to use for x, y, and z data.
        time : datetime
            Time in which to slice through objects.
        function : string
            Defaults to cubic function for interpolation.
            See scipy.interpolate.Rbf for additional options.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        grid_delta : 1D tuple, list, or array
            x and y deltas for creating grid.
        grid_buffer : float
            Buffer to apply to grid.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.contour`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        # Get x, y, and z data by looping through each dictionary
        # item and extracting data from appropriate time
        x = []
        y = []
        z = []
        for ds in self._arm:
            obj = self._arm[ds]
            field = fields[ds]
            x.append(obj[field[0]].sel(time=time).values.tolist())
            y.append(obj[field[1]].sel(time=time).values.tolist())
            z.append(obj[field[2]].sel(time=time).values.tolist())

        # Create a meshgrid for gridding onto
        xs = np.arange(np.min(x) - grid_buffer, np.max(x) + grid_buffer, grid_delta[0])
        ys = np.arange(np.min(y) - grid_buffer, np.max(y) + grid_buffer, grid_delta[1])
        xi, yi = np.meshgrid(xs, ys)

        # Use scipy radial basis function to interpolate data onto grid
        rbf = Rbf(x, y, z, function=function)
        zi = rbf(xi, yi)

        # Create contour plot
        self.contourf(xi, yi, zi, subplot_index=subplot_index, **kwargs)

        return self.axes[subplot_index]

    def contourf(self, x, y, z, subplot_index=(0,), **kwargs):
        """
        Base function for filled contours if user already has data gridded.
        If subplots have not been added yet, an axis will be created
        assuming that there is only going to be one plot.

        Parameters
        ----------
        x : array-like
            x coordinates or grid for z.
        y : array-like
            y coordinates or grid for z.
        z : array-like(x,y)
            Values over which to contour.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.contourf`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        self.axes[subplot_index].contourf(x, y, z, **kwargs)

        return self.axes[subplot_index]

    def contour(self, x, y, z, subplot_index=(0,), **kwargs):
        """
        Base function for contours if user already has data gridded.
        If subplots have not been added yet, an axis will be created
        assuming that there is only going to be one plot.

        Parameters
        ----------
        x : array-like
            x coordinates or grid for z.
        y : array-like
            y coordinates or grid for z.
        z : array-like(x, y)
            Values over which to contour.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.contour`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        self.axes[subplot_index].contour(x, y, z, **kwargs)

        return self.axes[subplot_index]

    def plot_vectors_from_spd_dir(self, fields, time=None, subplot_index=(0,),
                                  mesh=False, function='cubic',
                                  grid_delta=(0.01, 0.01),
                                  grid_buffer=0.1, **kwargs):
        """
        Extracts, grids, and creates a contour plot.
        If subplots have not been added yet, an axis will be created
        assuming that there is only going to be one plot.

        Parameters
        ----------
        fields : dict
            Dictionary of fields to use for x, y, and z data.
        time : datetime
            Time in which to slice through objects.
        mesh : boolean
            Set to True to interpolate u and v to grid and create wind barbs.
        function : string
            Defaults to cubic function for interpolation.
            See scipy.interpolate.Rbf for additional options.
        grid_delta : 1D tuple, list, or array
            x and y deltas for creating grid.
        grid_buffer : float
            Buffer to apply to grid.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.barbs`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        # Get x, y, and z data by looping through each dictionary
        # item and extracting data from appropriate time
        x = []
        y = []
        wspd = []
        wdir = []
        for ds in self._arm:
            obj = self._arm[ds]
            field = fields[ds]
            x.append(obj[field[0]].sel(time=time).values.tolist())
            y.append(obj[field[1]].sel(time=time).values.tolist())
            wspd.append(obj[field[2]].sel(time=time).values.tolist())
            wdir.append(obj[field[3]].sel(time=time).values.tolist())

        # Calculate u and v
        tempu = -np.sin(np.deg2rad(wdir)) * wspd
        tempv = -np.cos(np.deg2rad(wdir)) * wspd

        if mesh is True:
            # Create a meshgrid for gridding onto
            xs = np.arange(min(x) - grid_buffer, max(x) + grid_buffer, grid_delta[0])
            ys = np.arange(min(y) - grid_buffer, max(y) + grid_buffer, grid_delta[1])
            xi, yi = np.meshgrid(xs, ys)

            # Use scipy radial basis function to interpolate data onto grid
            rbf = Rbf(x, y, tempu, function=function)
            u = rbf(xi, yi)

            rbf = Rbf(x, y, tempv, function=function)
            v = rbf(xi, yi)
        else:
            xi = x
            yi = y
            u = tempu
            v = tempv

        self.barbs(xi, yi, u, v, **kwargs)

        return self.axes[subplot_index]

    def barbs(self, x, y, u, v, subplot_index=(0,), **kwargs):
        """
        Base function for wind barbs. If subplots have not been added yet,
        an axis will be created assuming that there is only going to be
        one plot.

        Parameters
        ----------
        x : array-like
            x coordinates or grid for z.
        y : array-like
            y coordinates or grid for z.
        u : array-like
            U component of vector.
        v : array-like
            V component of vector.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.barbs`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        self.axes[subplot_index].barbs(x, y, u, v, **kwargs)

        return self.axes[subplot_index]

    def plot_station(self, fields, time=None, subplot_index=(0,),
                     text_color='white', **kwargs):
        """
        Extracts, grids, and creates a contour plot. If subplots have not
        been added yet, an axis will be created assuming that there is only
        going to be one plot.

        Parameters
        ----------
        fields : dict
            Dictionary of fields to use for x, y, and z data.
        time : datetime
            Time in which to slice through objects.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.
        text_color : string
            Color for text.
        **kwargs : keyword arguments
            The keyword arguments for :func:`plt.plot`

        Returns
        -------
        ax : matplotlib axis handle
            The matplotlib axis handle of the plot.

        """
        # Get x, y, and data by looping through each dictionary
        # item and extracting data from appropriate time
        for ds in self._arm:
            obj = self._arm[ds]
            field = fields[ds]
            for i, f in enumerate(field):
                if i == 0:
                    x = obj[f].sel(time=time).values.tolist()
                elif i == 1:
                    y = obj[f].sel(time=time).values.tolist()
                    self.axes[subplot_index].plot(x, y, '*', **kwargs)
                else:
                    data = obj[f].sel(time=time).values.tolist()
                    offset = 0.02
                    if i == 2:
                        x1 = x - 3. * offset
                        y1 = y + offset
                    if i == 3:
                        x1 = x + offset
                        y1 = y + offset
                    if i == 4:
                        x1 = x + offset
                        y1 = y - 2. * offset
                    if i == 5:
                        x1 = x - 3. * offset
                        y1 = y - 2. * offset
                    if data < 5:
                        string = str(round(data, 1))
                    else:
                        string = str(round(data))
                    self.axes[subplot_index].text(x1, y1, string, color=text_color)

        return self.axes[subplot_index]
