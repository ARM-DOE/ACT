"""
Stores the class for SkewTDisplay.

"""

# Import third party libraries
import matplotlib.pyplot as plt
import numpy as np
import warnings

try:
    from pkg_resources import DistributionNotFound
    import metpy.calc as mpcalc
    METPY_AVAILABLE = True
except ImportError:
    METPY_AVAILABLE = False
except (ModuleNotFoundError, DistributionNotFound):
    warnings.warn("MetPy is installed but could not be imported. " +
                  "Please check your MetPy installation. Some features " +
                  "will be disabled.", ImportWarning)
    METPY_AVAILABLE = False

# Import Local Libs
from ..utils import datetime_utils as dt_utils
from copy import deepcopy
from .plot import Display

if METPY_AVAILABLE:
    from metpy.units import units
    from metpy.plots import SkewT


class SkewTDisplay(Display):
    """
    A class for making Skew-T plots.

    This is inherited from the :func:`act.plotting.Display`
    class and has therefore has the same attributes as that class.
    See :func:`act.plotting.Display`
    for more information. There are no additional attributes or parameters
    to this class.

    In order to create Skew-T plots, ACT needs the MetPy package to be
    installed on your system. More information about
    MetPy go here: https://unidata.github.io/MetPy/latest/index.html.

    Examples
    --------
    Here is an example of how to make a Skew-T plot using ACT:

    .. code-block :: python

        sonde_ds = act.io.armfiles.read_netcdf(
           act.tests.sample_files.EXAMPLE_SONDE1)

        skewt = act.plotting.SkewTDisplay(sonde_ds)
        skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
        plt.show()

    """
    def __init__(self, obj, subplot_shape=(1,), ds_name=None, **kwargs):
        # We want to use our routine to handle subplot adding, not the main
        # one
        if not METPY_AVAILABLE:
            raise ImportError("MetPy need to be installed on your system to " +
                              "make Skew-T plots.")
        new_kwargs = kwargs.copy()
        super().__init__(obj, None, ds_name,
                         subplot_kw=dict(projection='skewx'), **new_kwargs)

        # Make a SkewT object for each subplot
        self.add_subplots(subplot_shape, **kwargs)

    def add_subplots(self, subplot_shape=(1,), **kwargs):
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
                subplot_tuple = (subplot_shape[0], 1, i + 1)
                self.SkewT[i] = SkewT(fig=self.fig, subplot=subplot_tuple)
                self.axes[i] = self.SkewT[i].ax
        elif len(subplot_shape) == 2:
            for i in range(subplot_shape[0]):
                for j in range(subplot_shape[1]):
                    subplot_tuple = (subplot_shape[0],
                                     subplot_shape[1],
                                     i * subplot_shape[1] + j + 1)
                    self.SkewT[i, j] = SkewT(fig=self.fig, subplot=subplot_tuple)
                    self.axes[i, j] = self.SkewT[i, j].ax
        else:
            raise ValueError("Subplot shape must be 1 or 2D!")

    def set_xrng(self, xrng, subplot_index=(0,)):
        """
        Sets the x range of the plot.

        Parameters
        ----------
        xrng : 2 number array.
            The x limits of the plot.
        subplot_index : 1 or 2D tuple, list, or array
            The index of the subplot to set the x range of.

        """
        if self.axes is None:
            raise RuntimeError("set_xrng requires the plot to be displayed.")

        if not hasattr(self, 'xrng') or np.all(self.xrng == 0):
            if len(self.axes.shape) == 2:
                self.xrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2))
            else:
                self.xrng = np.zeros((self.axes.shape[0], 2))

        self.axes[subplot_index].set_xlim(xrng)
        self.xrng[subplot_index, :] = np.array(xrng)

    def set_yrng(self, yrng, subplot_index=(0,)):
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

        if not hasattr(self, 'yrng') or np.all(self.yrng == 0):
            if len(self.axes.shape) == 2:
                self.yrng = np.zeros((self.axes.shape[0], self.axes.shape[1], 2))
            else:
                self.yrng = np.zeros((self.axes.shape[0], 2))

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
        spd_field : str
            The name of the field corresponding to the wind speed.
        dir_field : str
            The name of the field corresponding to the wind direction
            in degrees from North.
        p_field : str
            The name of the field containing the atmospheric pressure.
        t_field : str
            The name of the field containing the atmospheric temperature.
        td_field : str
            The name of the field containing the dewpoint.
        dsname : str or None
            The name of the datastream to plot. Set to None to make ACT
            attempt to automatically determine this.
        kwargs : dict
            Additional keyword arguments will be passed into
            :func:`act.plotting.SkewTDisplay.plot_from_u_and_v`

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

        # Make temporary field called tempu, tempv
        spd = self._obj[dsname][spd_field].values
        dir = self._obj[dsname][dir_field].values
        tempu = -np.sin(np.deg2rad(dir)) * spd
        tempv = -np.cos(np.deg2rad(dir)) * spd
        self._obj[dsname]["temp_u"] = deepcopy(self._obj[dsname][spd_field])
        self._obj[dsname]["temp_v"] = deepcopy(self._obj[dsname][spd_field])
        self._obj[dsname]["temp_u"].values = tempu
        self._obj[dsname]["temp_v"].values = tempv
        the_ax = self.plot_from_u_and_v("temp_u", "temp_v", p_field,
                                        t_field, td_field, dsname, **kwargs)
        del self._obj[dsname]["temp_u"], self._obj[dsname]["temp_v"]
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
        u_field : str
            The name of the field containing the u component of the wind.
        v_field : str
            The name of the field containing the v component of the wind.
        p_field : str
            The name of the field containing the pressure.
        t_field : str
            The name of the field containing the temperature.
        td_field : str
            The name of the field containing the dewpoint temperature.
        dsname : str or None
            The name of the datastream to plot. Set to None to make ACT
            attempt to automatically determine this.
        subplot_index : tuple
            The index of the subplot to make the plot on.
        p_levels_to_plot : 1D array
            The pressure levels to plot the wind barbs on. Set to None
            to have ACT to use neatly spaced defaults of
            50, 100, 200, 300, 400, 500, 600, 700, 750, 800,
            850, 900, 950, and 1000 hPa.
        show_parcel : bool
            Set to True to show the temperature of a parcel lifted
            from the surface.
        shade_cape : bool
            Set to True to shade the CAPE red.
        shade_cin : bool
            Set to True to shade the CIN blue.
        set_title : None or str
            The title of the plot is set to this. Set to None to use
            a default title.
        plot_barbs_kwargs : dict
            Additional keyword arguments to pass into MetPy's
            SkewT.plot_barbs.
        plot_kwargs : dict
            Additional keyword arguments to pass into MetPy's
            SkewT.plot.

        Returns
        -------
        ax : matplotlib axis handle
            The axis handle to the plot.

        """
        if dsname is None and len(self._obj.keys()) > 1:
            raise ValueError(("You must choose a datastream when there are 2 "
                              "or more datasets in the TimeSeriesDisplay "
                              "object."))
        elif dsname is None:
            dsname = list(self._obj.keys())[0]

        if p_levels_to_plot is None:
            p_levels_to_plot = np.array([50., 100., 200., 300., 400.,
                                         500., 600., 700., 750., 800.,
                                         850., 900., 950., 1000.])
        T = self._obj[dsname][t_field]
        T_units = self._obj[dsname][t_field].attrs["units"]
        if T_units == "C":
            T_units = "degC"

        T = T.values * getattr(units, T_units)
        Td = self._obj[dsname][td_field]
        Td_units = self._obj[dsname][td_field].attrs["units"]
        if Td_units == "C":
            Td_units = "degC"

        Td = Td.values * getattr(units, Td_units)
        u = self._obj[dsname][u_field]
        u_units = self._obj[dsname][u_field].attrs["units"]
        u = u.values * getattr(units, u_units)

        v = self._obj[dsname][v_field]
        v_units = self._obj[dsname][v_field].attrs["units"]
        v = v.values * getattr(units, v_units)

        p = self._obj[dsname][p_field]
        p_units = self._obj[dsname][p_field].attrs["units"]
        p = p.values * getattr(units, p_units)

        u_red = np.zeros_like(p_levels_to_plot) * getattr(units, u_units)
        v_red = np.zeros_like(p_levels_to_plot) * getattr(units, v_units)
        p_levels_to_plot = p_levels_to_plot * getattr(units, p_units)
        for i in range(len(p_levels_to_plot)):
            index = np.argmin(np.abs(p_levels_to_plot[i] - p))
            u_red[i] = u[index].magnitude * getattr(units, u_units)
            v_red[i] = v[index].magnitude * getattr(units, v_units)

        u_red = u_red.magnitude
        v_red = v_red.magnitude
        self.SkewT[subplot_index].plot(p, T, 'r', **plot_kwargs)
        self.SkewT[subplot_index].plot(p, Td, 'g', **plot_kwargs)
        self.SkewT[subplot_index].plot_barbs(
            p_levels_to_plot, u_red, v_red, **plot_barbs_kwargs)

        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
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
                 dt_utils.numpy_to_arm_date(self._obj[dsname].time.values[0])])

        self.axes[subplot_index].set_title(set_title)

        # Set Y Limit
        our_data = p.magnitude
        if np.isfinite(our_data).any():
            yrng = [np.nanmax(our_data), np.nanmin(our_data)]
        else:
            yrng = [1000., 100.]
        self.set_yrng(yrng, subplot_index)

        # Set X Limit
        xrng = [np.nanmin(T.magnitude) - 10., np.nanmax(T.magnitude) + 10.]
        self.set_xrng(xrng, subplot_index)

        return self.axes[subplot_index]
