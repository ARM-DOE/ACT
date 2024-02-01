"""
Stores the class for SkewTDisplay.

"""

from copy import deepcopy

import matplotlib.pyplot as plt

# Import third party libraries
import metpy
import metpy.calc as mpcalc
import numpy as np
import scipy
from metpy.plots import Hodograph, SkewT
from metpy.units import units

from ..retrievals import calculate_stability_indicies

# Import Local Libs
from ..utils import datetime_utils as dt_utils
from .plot import Display


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

        sonde_ds = act.io.arm.read_arm_netcdf(
           act.tests.sample_files.EXAMPLE_SONDE1)

        skewt = act.plotting.SkewTDisplay(sonde_ds)
        skewt.plot_from_u_and_v('u_wind', 'v_wind', 'pres', 'tdry', 'dp')
        plt.show()

    """

    def __init__(self, ds, subplot_shape=(1,), subplot=None, ds_name=None, set_fig=None, **kwargs):
        # We want to use our routine to handle subplot adding, not the main
        # one
        new_kwargs = kwargs.copy()
        super().__init__(ds, None, ds_name, subplot_kw=dict(projection='skewx'), **new_kwargs)

        # Make a SkewT object for each subplot
        self.add_subplots(subplot_shape, set_fig=set_fig, subplot=subplot, **kwargs)

    def add_subplots(self, subplot_shape=(1,), set_fig=None, subplot=None, **kwargs):
        """
        Adds subplots to the Display object. The current
        figure in the object will be deleted and overwritten.

        Parameters
        ----------
        subplot_shape : 1 or 2D tuple, list, or array
            The structure of the subplots in (rows, cols).
        subplot_kw : dict, optional
            The kwargs to pass into fig.subplots.
        set_fig : matplotlib figure, optional
            Figure to pass to SkewT
        **kwargs : keyword arguments
            Any other keyword arguments that will be passed
            into :func:`matplotlib.pyplot.figure` when the figure
            is made. The figure is only made if the *fig*
            property is None. See the matplotlib
            documentation for further details on what keyword
            arguments are available.

        """
        del self.axes
        if self.fig is None and set_fig is None:
            self.fig = plt.figure(**kwargs)
        if set_fig is not None:
            self.fig = set_fig
        self.SkewT = np.empty(shape=subplot_shape, dtype=SkewT)
        self.axes = np.empty(shape=subplot_shape, dtype=plt.Axes)
        if len(subplot_shape) == 1:
            for i in range(subplot_shape[0]):
                if subplot is None:
                    subplot_tuple = (subplot_shape[0], 1, i + 1)
                else:
                    subplot_tuple = subplot
                self.SkewT[i] = SkewT(fig=self.fig, subplot=subplot_tuple)
                self.axes[i] = self.SkewT[i].ax
        elif len(subplot_shape) == 2:
            for i in range(subplot_shape[0]):
                for j in range(subplot_shape[1]):
                    subplot_tuple = (
                        subplot_shape[0],
                        subplot_shape[1],
                        i * subplot_shape[1] + j + 1,
                    )
                    self.SkewT[i, j] = SkewT(fig=self.fig, subplot=subplot_tuple)
                    self.axes[i, j] = self.SkewT[i, j].ax
        else:
            raise ValueError('Subplot shape must be 1 or 2D!')

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
            raise RuntimeError('set_xrng requires the plot to be displayed.')

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
            raise RuntimeError('set_yrng requires the plot to be displayed.')

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

    def plot_from_spd_and_dir(
        self, spd_field, dir_field, p_field, t_field, td_field, dsname=None, **kwargs
    ):
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
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Make temporary field called tempu, tempv
        spd = self._ds[dsname][spd_field].values * units(self._ds[dsname][spd_field].attrs['units'])
        dir = self._ds[dsname][dir_field].values * units(self._ds[dsname][dir_field].attrs['units'])
        tempu, tempv = mpcalc.wind_components(spd, dir)

        self._ds[dsname]['temp_u'] = deepcopy(self._ds[dsname][spd_field])
        self._ds[dsname]['temp_v'] = deepcopy(self._ds[dsname][spd_field])
        self._ds[dsname]['temp_u'].values = tempu
        self._ds[dsname]['temp_v'].values = tempv
        the_ax = self.plot_from_u_and_v(
            'temp_u', 'temp_v', p_field, t_field, td_field, dsname, **kwargs
        )
        del self._ds[dsname]['temp_u'], self._ds[dsname]['temp_v']
        return the_ax

    def plot_from_u_and_v(
        self,
        u_field,
        v_field,
        p_field,
        t_field,
        td_field,
        dsname=None,
        subplot_index=(0,),
        p_levels_to_plot=None,
        show_parcel=True,
        shade_cape=True,
        shade_cin=True,
        set_title=None,
        smooth_p=3,
        plot_dry_adiabats=False,
        plot_moist_adiabats=False,
        plot_mixing_lines=False,
        plot_barbs_kwargs=dict(),
        plot_kwargs=dict(),
        dry_adiabats_kwargs=dict(),
        moist_adiabats_kwargs=dict(),
        mixing_lines_kwargs=dict(),
    ):
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
            25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 750, 800,
            850, 900, 950, and 1000 hPa.
        show_parcel : bool
            Set to true to calculate the profile a parcel takes through the atmosphere
            using the metpy.calc.parcel_profile function.  From their documentation,
            the parcel starts at the surface temperature and dewpoint, is lifted up
            dry adiabatically to the LCL and then moist adiabatically from there.
        shade_cape : bool
            Set to True to shade the CAPE red.
        shade_cin : bool
            Set to True to shade the CIN blue.
        set_title : None or str
            The title of the plot is set to this. Set to None to use
            a default title.
        smooth_p : int
            If pressure is not in descending order, will smooth the data
            using this many points to try and work around the issue.
            Default is 3 but inthe pbl retrieval code we have to default to 5 at times
        plot_barbs_kwargs : dict
            Additional keyword arguments to pass into MetPy's
            SkewT.plot_barbs.
        plot_kwargs : dict
            Additional keyword arguments to pass into MetPy's
            SkewT.plot.
        dry_adiabats_kwargs : dict
            Additional keyword arguments to pass into MetPy's plot_dry_adiabats function
        moist_adiabats_kwargs : dict
            Additional keyword arguments to pass into MetPy's plot_moist_adiabats function
        mixing_lines_kwargs : dict
            Additional keyword arguments to pass into MetPy's plot_mixing_lines function

        Returns
        -------
        ax : matplotlib axis handle
            The axis handle to the plot.

        References
        ----------
        May, R. M., Arms, S. C., Marsh, P., Bruning, E., Leeman, J. R., Goebbert, K., Thielen, J. E.,
            Bruick, Z., and Camron, M. D., 2023: MetPy: A Python Package for Meteorological Data.
            Unidata, Unidata/MetPy, doi:10.5065/D6WW7G29.

        """
        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        if p_levels_to_plot is None:
            p_levels_to_plot = np.array(
                [
                    25.0,
                    50.0,
                    75.0,
                    100.0,
                    150.0,
                    200.0,
                    250.0,
                    300.0,
                    400.0,
                    500.0,
                    600.0,
                    700.0,
                    750.0,
                    800.0,
                    850.0,
                    900.0,
                    950.0,
                    1000.0,
                ]
            ) * units('hPa')

        # Get pressure and smooth if not in order
        p = self._ds[dsname][p_field]
        if not all(p[i] <= p[i + 1] for i in range(len(p) - 1)):
            if 'time' in self._ds:
                self._ds[dsname][p_field] = (
                    self._ds[dsname][p_field]
                    .rolling(time=smooth_p, min_periods=1, center=True)
                    .mean()
                )
                p = self._ds[dsname][p_field]

        p_units = self._ds[dsname][p_field].attrs['units']
        p = p.values * getattr(units, p_units)
        if len(np.shape(p)) == 2:
            p = np.reshape(p, p.shape[0] * p.shape[1])

        T = self._ds[dsname][t_field]
        T_units = self._ds[dsname][t_field].attrs['units']
        if T_units == 'C':
            T_units = 'degC'

        T = T.values * getattr(units, T_units)
        if len(np.shape(T)) == 2:
            T = np.reshape(T, T.shape[0] * T.shape[1])

        Td = self._ds[dsname][td_field]
        Td_units = self._ds[dsname][td_field].attrs['units']
        if Td_units == 'C':
            Td_units = 'degC'

        Td = Td.values * getattr(units, Td_units)
        if len(np.shape(Td)) == 2:
            Td = np.reshape(Td, Td.shape[0] * Td.shape[1])

        u = self._ds[dsname][u_field]
        u_units = self._ds[dsname][u_field].attrs['units']
        u = u.values * getattr(units, u_units)
        if len(np.shape(u)) == 2:
            u = np.reshape(u, u.shape[0] * u.shape[1])

        v = self._ds[dsname][v_field]
        v_units = self._ds[dsname][v_field].attrs['units']
        v = v.values * getattr(units, v_units)
        if len(np.shape(v)) == 2:
            v = np.reshape(v, v.shape[0] * v.shape[1])

        u_red = np.zeros_like(p_levels_to_plot) * getattr(units, u_units)
        v_red = np.zeros_like(p_levels_to_plot) * getattr(units, v_units)

        # Check p_levels_to_plot units, and convert to p units if needed
        if not hasattr(p_levels_to_plot, 'units'):
            p_levels_to_plot = p_levels_to_plot * getattr(units, p_units)
        else:
            p_levels_to_plot = p_levels_to_plot.to(p_units)

        for i in range(len(p_levels_to_plot)):
            index = np.argmin(np.abs(p_levels_to_plot[i] - p))
            u_red[i] = u[index].magnitude * getattr(units, u_units)
            v_red[i] = v[index].magnitude * getattr(units, v_units)

        self.SkewT[subplot_index].plot(p, T, 'r', **plot_kwargs)
        self.SkewT[subplot_index].plot(p, Td, 'g', **plot_kwargs)
        self.SkewT[subplot_index].plot_barbs(
            p_levels_to_plot.magnitude, u_red, v_red, **plot_barbs_kwargs
        )

        # Metpy fix if Pressure does not decrease monotonically in
        # your sounding.
        try:
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        except metpy.calc.exceptions.InvalidSoundingError:
            p = scipy.ndimage.median_filter(p, 3, output=float)
            p = metpy.units.units.Quantity(p, p_units)
            prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')

        if show_parcel:
            # Only plot where prof > T
            lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
            self.SkewT[subplot_index].plot(
                lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black', **plot_kwargs
            )
            self.SkewT[subplot_index].plot(p, prof, 'k', linewidth=2, **plot_kwargs)

        if shade_cape:
            self.SkewT[subplot_index].shade_cape(p, T, prof, linewidth=2)

        if shade_cin:
            self.SkewT[subplot_index].shade_cin(p, T, prof, linewidth=2)

        # Get plot temperatures from x-axis as t0
        t0 = self.SkewT[subplot_index].ax.get_xticks() * getattr(units, T_units)

        # Add minimum pressure to pressure levels to plot
        if np.nanmin(p.magnitude) < np.nanmin(p_levels_to_plot.magnitude):
            plp = np.insert(p_levels_to_plot.magnitude, 0, np.nanmin(p.magnitude)) * units('hPa')
        else:
            plp = p_levels_to_plot

        # New options for plotting dry and moist adiabats as well as the mixing lines
        if plot_dry_adiabats:
            self.SkewT[subplot_index].plot_dry_adiabats(pressure=plp, t0=t0, **dry_adiabats_kwargs)

        if plot_moist_adiabats:
            self.SkewT[subplot_index].plot_moist_adiabats(
                t0=t0, pressure=plp, **moist_adiabats_kwargs
            )

        if plot_mixing_lines:
            self.SkewT[subplot_index].plot_mixing_lines(pressure=plp, **mixing_lines_kwargs)

        # Set Title
        if set_title is None:
            if 'time' in self._ds[dsname]:
                title_time = (dt_utils.numpy_to_arm_date(self._ds[dsname].time.values[0]),)
            elif '_file_dates' in self._ds[dsname].attrs:
                title_time = self._ds[dsname].attrs['_file_dates'][0]
            else:
                title_time = ''
            set_title = ' '.join([dsname, 'on', title_time[0]])

        self.axes[subplot_index].set_title(set_title)

        # Set Y Limit
        our_data = p.magnitude
        if np.isfinite(our_data).any():
            yrng = [np.nanmax(our_data), np.nanmin(our_data)]
        else:
            yrng = [1000.0, 100.0]
        self.set_yrng(yrng, subplot_index)

        # Set X Limit
        xrng = [np.nanmin(T.magnitude) - 10.0, np.nanmax(T.magnitude) + 10.0]
        self.set_xrng(xrng, subplot_index)

        return self.axes[subplot_index]

    def plot_hodograph(
        self,
        spd_field,
        dir_field,
        color_field=None,
        set_fig=None,
        set_axes=None,
        component_range=80,
        dsname=None,
        uv_flag=False,
    ):
        """
        This will plot a hodograph from the radiosonde wind data using
        MetPy

        Parameters
        ----------
        spd_field : str
            The name of the field corresponding to the wind speed.
        dir_field : str
            The name of the field corresponding to the wind direction
            in degrees from North.
        color_field : str, optional
            The name of the field if wanting to shade by another variable
        set_fig : matplotlib figure, optional
            The figure to plot on
        set_axes : matplotlib axes, optional
            The specific axes to plot on
        component_range : int
             Range of the hodograph.  Default is 80
        dsname : str
             Name of the datastream to plot if multiple in the plot object
        uv_flag : boolean
             If set to True, spd_field and dir_field will be treated as the
             U and V wind variable names

        Returns
        -------
        self.axes : matplotlib axes

        """

        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Get the current plotting axis
        if set_fig is not None:
            self.fig = set_fig
        if set_axes is not None:
            self.axes = set_axes

        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        # Calculate u/v wind components from speed/direction
        if uv_flag is False:
            spd = self._ds[dsname][spd_field].values * units(
                self._ds[dsname][spd_field].attrs['units']
            )
            dir = self._ds[dsname][dir_field].values * units(
                self._ds[dsname][dir_field].attrs['units']
            )
            u, v = mpcalc.wind_components(spd, dir)
        else:
            u = self._ds[dsname][spd_field].values * units(
                self._ds[dsname][spd_field].attrs['units']
            )
            v = self._ds[dsname][dir_field].values * units(
                self._ds[dsname][dir_field].attrs['units']
            )

        # Plot out the data using the Hodograph method
        h = Hodograph(self.axes, component_range=component_range)
        h.add_grid(increment=20)
        if color_field is None:
            h.plot(u, v)
        else:
            data = self._ds[dsname][color_field].values * units(
                self._ds[dsname][color_field].attrs['units']
            )
            h.plot_colormapped(u, v, data)

        return self.axes

    def add_stability_info(
        self,
        temp_name='tdry',
        td_name='dp',
        p_name='pres',
        overwrite_data=None,
        add_data=None,
        set_fig=None,
        set_axes=None,
        dsname=None,
    ):
        """
        This plot will make a sounding plot from wind data that is given
        in speed and direction.

        Parameters
        ----------
        temp_name : str
            The name of the temperature field.
        td_name : str
            The name of the dewpoint field.
        p_name : str
            The name of the pressure field.
        overwrite_data : dict
            A disctionary of variables/values to write out instead
            of the ones calculated by MetPy.  Needs to be of the form
            .. code-block:: python

                overwrite_data={'LCL': 234, 'CAPE': 25}
                ...
        add_data : dict
            A dictionary of variables and values to write out in
            addition to the MetPy calculated ones
        set_fig : matplotlib figure, optional
            The figure to plot on
        set_axes : matplotlib axes, optional
            The specific axes to plot on
        dsname : str
             Name of the datastream to plot if multiple in the plot object

        Returns
        -------
        self.axes : matplotlib axes

        """

        if dsname is None and len(self._ds.keys()) > 1:
            raise ValueError(
                'You must choose a datastream when there are 2 '
                'or more datasets in the TimeSeriesDisplay '
                'object.'
            )
        elif dsname is None:
            dsname = list(self._ds.keys())[0]

        # Get the current plotting axis
        if set_fig is not None:
            self.fig = set_fig
        if set_axes is not None:
            self.axes = set_axes

        if self.fig is None:
            self.fig = plt.figure()

        if self.axes is None:
            self.axes = np.array([plt.axes()])
            self.fig.add_axes(self.axes[0])

        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        self.axes.get_xaxis().set_ticks([])
        self.axes.get_yaxis().set_ticks([])
        ct = 0
        if overwrite_data is None:
            # Calculate stability indicies
            ds_sonde = calculate_stability_indicies(
                self._ds[dsname],
                temp_name=temp_name,
                td_name=td_name,
                p_name=p_name,
            )

            # Add MetPy calculated variables to the list
            variables = {
                'lifted_index': 'Lifted Index',
                'surface_based_cape': 'SBCAPE',
                'surface_based_cin': 'SBCIN',
                'most_unstable_cape': 'MUCAPE',
                'most_unstable_cin': 'MUCIN',
                'lifted_condensation_level_temperature': 'LCL Temp',
                'lifted_condensation_level_pressure': 'LCL Pres',
            }
            for i, v in enumerate(variables):
                var_string = str(np.round(ds_sonde[v].values, 2))
                self.axes.text(
                    -0.05,
                    (0.98 - (0.1 * i)),
                    variables[v] + ': ',
                    transform=self.axes.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                )
                self.axes.text(
                    0.95,
                    (0.98 - (0.1 * i)),
                    var_string,
                    transform=self.axes.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                )
                ct += 1
        else:
            # If overwrite_data is set, the user passes in their own dictionary
            for i, v in enumerate(overwrite_data):
                var_string = str(np.round(overwrite_data[v], 2))
                self.axes.text(
                    -0.05,
                    (0.98 - (0.1 * i)),
                    v + ': ',
                    transform=self.axes.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                )
                self.axes.text(
                    0.95,
                    (0.98 - (0.1 * i)),
                    var_string,
                    transform=self.axes.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                )
        # User can also add variables to the existing ones calculated by MetPy
        if add_data is not None:
            for i, v in enumerate(add_data):
                var_string = str(np.round(add_data[v], 2))
                self.axes.text(
                    -0.05,
                    (0.98 - (0.1 * (i + ct))),
                    v + ': ',
                    transform=self.axes.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                )
                self.axes.text(
                    0.95,
                    (0.98 - (0.1 * (i + ct))),
                    var_string,
                    transform=self.axes.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='right',
                )
        return self.axes

    def plot_enhanced_skewt(
        self,
        spd_name='wspd',
        dir_name='deg',
        temp_name='tdry',
        td_name='dp',
        p_name='pres',
        overwrite_data=None,
        add_data=None,
        color_field=None,
        component_range=80,
        uv_flag=False,
        dsname=None,
        figsize=(14, 10),
        layout='constrained',
    ):
        """
        This will plot an enhanced Skew-T plot with a Hodograph on the top right
        and the stability parameters on the lower right.  This will create a new
        figure so that one does not need to be defined through subplot_shape.

        Requires Matplotlib v 3.7 and higher

        Parameters
        ----------
        spd_name : str
            The name of the field corresponding to the wind speed.
        dir_name : str
            The name of the field corresponding to the wind direction
            in degrees from North.
        temp_name : str
            The name of the temperature field.
        td_name : str
            The name of the dewpoint field.
        p_name : str
            The name of the pressure field.
        overwrite_data : dict
            A disctionary of variables/values to write out instead
            of the ones calculated by MetPy.  Needs to be of the form
            .. code-block:: python

                overwrite_data={'LCL': 234, 'CAPE': 25}
                ...
        add_data : dict
            A dictionary of variables and values to write out in
            addition to the MetPy calculated ones
        color_field : str, optional
            The name of the field if wanting to shade by another variable
        component_range : int
             Range of the hodograph.  Default is 80
        uv_flag : boolean
             If set to True, spd_field and dir_field will be treated as the
             U and V wind variable names
        dsname : str
             Name of the datastream to plot if multiple in the plot object
        figsize : tuple
             Figure size for the plot
        layout : str
            String to pass to matplotlib.figure.Figure object layout keyword
            argument. Choice of 'constrained,' 'compressed,' 'tight,' or None.
            Default is 'constrained'.

        Returns
        -------
        self.axes : matplotlib axes

        """

        # Set up the figure and axes
        # Close existing figure as a new one will be created
        plt.close('all')
        subplot_kw = {'a': {'projection': 'skewx'}}
        fig, axs = plt.subplot_mosaic(
            [['a', 'a', 'b'], ['a', 'a', 'b'], ['a', 'a', 'c'], ['a', 'a', 'c']],
            layout=layout,
            per_subplot_kw=subplot_kw,
        )
        self.fig = fig
        self.axes = axs

        # Plot out the Skew-T
        display = SkewTDisplay(self._ds, set_fig=fig, subplot=axs['a'], figsize=figsize)
        if uv_flag is True:
            display.plot_from_u_and_v(spd_name, dir_name, p_name, temp_name, td_name)
        else:
            display.plot_from_spd_and_dir(spd_name, dir_name, p_name, temp_name, td_name)

        # Plot the hodograph
        display.plot_hodograph(
            spd_name,
            dir_name,
            set_axes=axs['b'],
            color_field=color_field,
            component_range=component_range,
            dsname=dsname,
            uv_flag=uv_flag,
        )

        # Add Stability information
        display.add_stability_info(
            set_axes=axs['c'],
            temp_name=temp_name,
            td_name=td_name,
            p_name=p_name,
            overwrite_data=overwrite_data,
            add_data=add_data,
            dsname=dsname,
        )
        return self.axes
