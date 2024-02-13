"""
ARM Data Surveyor (ADS)
Command line wrapper around ACT.  Not all
features of ACT are included as options in ADS.
Please see the examples.txt for examples on how
to use ADS.

Author: Jason Hemedinger

"""

import argparse
import json
import glob
import ast
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import act

try:
    import cartopy.crs as ccrs

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


def option_error_check(args, error_fields, check_all=False):
    '''
    This will check the args object for keys to see if they are set. If
    at least one key is not set will or all keys wiht check_all set
    will print error message and exit.
    '''

    if not isinstance(error_fields, (list, tuple)):
        error_fields = [error_fields]

    print_error = False
    how_many = 'one'
    if check_all is False and not any([vars(args)[ii] for ii in error_fields]):
        print_error = True

    if check_all is True and not all([vars(args)[ii] for ii in error_fields]):
        print_error = True
        how_many = 'all'

    if print_error:
        prepend = '--'
        for ii, value in enumerate(error_fields):
            if not value.startswith(prepend):
                error_fields[ii] = prepend + value

        print(
            f"\n{pathlib.Path(__file__).name}: error: {how_many} of the arguments "
            f"{' '.join(error_fields)} is requried\n"
        )
        exit()


def find_drop_vars(args):
    '''
    This will check if more than one file is to be read. If so read one file
    and get list of variables to not read based on the fields arguments and
    corresponding QC or dimention variables. This will significantly speed up
    the reading time for reading many files.
    '''
    files = glob.glob(args.file_path)
    drop_vars = []
    if len(files) > 1:
        ds = act.io.arm.read_arm_netcdf(files[0])
        ds.clean.cleanup()
        drop_vars = set(ds.data_vars)
        keep_vars = ['latitude', 'longitude']
        if args.field is not None:
            keep_vars.append(args.field)

        if args.fields is not None:
            keep_vars.extend(set(args.fields))

        if args.wind_fields is not None:
            keep_vars.extend(set(args.wind_fields))

        if args.station_fields is not None:
            keep_vars.extend(set(args.station_fields))

        if args.latitude is not None:
            keep_vars.append(args.latitude)

        if args.longitude is not None:
            keep_vars.append(args.longitude)

        if args.x_field is not None:
            keep_vars.append(args.x_field)

        if args.y_field is not None:
            keep_vars.append(args.y_field)

        if args.u_wind is not None:
            keep_vars.append(args.u_wind)

        if args.v_wind is not None:
            keep_vars.append(args.v_wind)

        if args.p_field is not None:
            keep_vars.append(args.p_field)

        if args.t_field is not None:
            keep_vars.append(args.t_field)

        if args.td_field is not None:
            keep_vars.append(args.td_field)

        if args.spd_field is not None:
            keep_vars.append(args.spd_field)

        if args.dir_field is not None:
            keep_vars.append(args.dir_field)

        keep_vars_additional = []
        for var_name in keep_vars:
            qc_var_name = ds.qcfilter.check_for_ancillary_qc(
                var_name, add_if_missing=False, cleanup=False
            )
            if qc_var_name is not None:
                keep_vars_additional.append(qc_var_name)

            try:
                keep_vars_additional.extend(ds[var_name].dims)
            except KeyError:
                pass

        drop_vars = drop_vars - set(keep_vars) - set(keep_vars_additional)

    return drop_vars


def geodisplay(args):
    ds = act.io.arm.read_arm_netcdf(args.file_path)

    dsname = args.dsname
    if dsname == _default_dsname:
        try:
            dsname = ds.attrs['datastream']
        except KeyError:
            pass

    display = act.plotting.GeographicPlotDisplay({dsname: ds}, figsize=args.figsize)

    display.geoplot(
        data_field=args.field,
        lat_field=args.latitude,
        lon_field=args.longitude,
        dsname=dsname,
        cbar_label=args.cb_label,
        title=args.set_title,
        plot_buffer=args.plot_buffer,
        stamen=args.stamen,
        tile=args.tile,
        cartopy_feature=args.cfeatures,
        cmap=args.cmap,
        text=args.text,
        gridlines=args.gridlines,
        projection=args.projection,
        **args.kwargs,
    )

    plt.savefig(args.out_path)
    plt.show()
    plt.close(display.fig)

    ds.close()


def skewt(args):
    ds = act.io.arm.read_arm_netcdf(args.file_path)

    subplot_index = args.subplot_index

    dsname = args.dsname
    if dsname == _default_dsname:
        try:
            dsname = ds.attrs['datastream']
        except KeyError:
            pass

    display = act.plotting.SkewTDisplay({dsname: ds}, figsize=args.figsize)

    if args.from_u_and_v:
        display.plot_from_u_and_v(
            u_field=args.u_wind,
            v_field=args.v_wind,
            p_field=args.p_field,
            t_field=args.t_field,
            td_field=args.td_field,
            subplot_index=subplot_index,
            dsname=dsname,
            show_parcel=args.show_parcel,
            p_levels_to_plot=args.plevels_plot,
            shade_cape=args.shade_cape,
            shade_cin=args.shade_cin,
            set_title=args.set_title,
            plot_barbs_kwargs=args.plot_barbs_kwargs,
            plot_kwargs=args.plot_kwargs,
        )

    if args.from_spd_and_dir:
        display.plot_from_spd_and_dir(
            spd_field=args.spd_field,
            dir_field=args.dir_field,
            p_field=args.p_field,
            t_field=args.t_field,
            td_field=args.td_field,
            dsname=dsname,
            **args.kwargs,
        )

    plt.savefig(args.out_path)
    plt.show()
    plt.close(display.fig)

    ds.close()


def xsection(args):
    ds = act.io.arm.read_arm_netcdf(args.file_path)

    subplot_index = args.subplot_index

    dsname = args.dsname
    if dsname == _default_dsname:
        try:
            dsname = ds.attrs['datastream']
        except KeyError:
            pass

    display = act.plotting.XSectionDisplay({dsname: ds}, figsize=args.figsize)

    if args.plot_xsection:
        display.plot_xsection(
            dsname=dsname,
            varname=args.field,
            x=args.x_field,
            y=args.y_field,
            subplot_index=subplot_index,
            sel_kwargs=args.sel_kwargs,
            isel_kwargs=args.isel_kwargs,
            **args.kwargs,
        )

    if args.xsection_map:
        display.plot_xsection_map(
            dsname=dsname,
            varname=args.field,
            subplot_index=subplot_index,
            coastlines=args.coastlines,
            background=args.background,
            **args.kwargs,
        )

    plt.savefig(args.out_path)
    plt.show()
    plt.close(display.fig)

    ds.close()


def wind_rose(args):
    drop_vars = find_drop_vars(args)

    ds = act.io.arm.read_arm_netcdf(args.file_path, drop_variables=drop_vars)

    subplot_index = args.subplot_index

    dsname = args.dsname
    if dsname == _default_dsname:
        try:
            dsname = ds.attrs['datastream']
        except KeyError:
            pass

    display = act.plotting.WindRoseDisplay({dsname: ds}, figsize=args.figsize)

    display.plot(
        dir_field=args.dir_field,
        spd_field=args.spd_field,
        subplot_index=subplot_index,
        dsname=dsname,
        cmap=args.cmap,
        set_title=args.set_title,
        num_dirs=args.num_dir,
        spd_bins=args.spd_bins,
        tick_interval=args.tick_interval,
        **args.kwargs,
    )
    plt.savefig(args.out_path)
    plt.show()
    plt.close(display.fig)

    ds.close()


def timeseries(args):
    drop_vars = find_drop_vars(args)

    ds = act.io.arm.read_arm_netcdf(args.file_path, drop_variables=drop_vars)

    if args.cleanup:
        ds.clean.cleanup()

    subplot_shape = args.subplot_shape
    subplot_index = args.subplot_index

    dsname = args.dsname
    if dsname == _default_dsname:
        try:
            dsname = ds.attrs['datastream']
        except KeyError:
            pass

    display = act.plotting.TimeSeriesDisplay(
        {dsname: ds}, figsize=args.figsize, subplot_shape=subplot_shape
    )

    options = [
        'plot',
        'barbs_spd_dir',
        'barbs_u_v',
        'xsection_from_1d',
        'time_height_scatter',
        'qc',
        'fill_between',
        'multi_panel',
    ]
    option_error_check(args, options)

    if args.plot:
        option_error_check(args, 'field')
        if args.set_yrange is not None:
            yrange = list(map(float, args.set_yrange))
        else:
            yrange = args.set_yrange
        display.plot(
            field=args.field,
            dsname=dsname,
            cmap=args.cmap,
            set_title=args.set_title,
            add_nan=args.add_nan,
            subplot_index=subplot_index,
            use_var_for_y=args.var_y,
            day_night_background=args.day_night,
            invert_y_axis=args.invert_y_axis,
            abs_limits=args.abs_limits,
            time_rng=args.time_rng,
            assessment_overplot=args.assessment_overplot,
            assessment_overplot_category=args.overplot_category,
            assessment_overplot_category_color=args.category_color,
            force_line_plot=args.force_line_plot,
            labels=args.labels,
            cbar_label=args.cb_label,
            secondary_y=args.secondary_y,
            y_rng=yrange,
            **args.kwargs,
        )

    if args.barbs_spd_dir:
        display.plot_barbs_from_spd_dir(
            dir_field=args.dir_field,
            spd_field=args.spd_field,
            pres_field=args.p_field,
            dsname=dsname,
            **args.kwargs,
        )

    if args.barbs_u_v:
        display.plot_barbs_from_u_v(
            u_field=args.u_wind,
            v_field=args.v_wind,
            pres_field=args.p_field,
            dsname=dsname,
            set_title=args.set_title,
            invert_y_axis=args.invert_y_axis,
            day_night_background=args.day_night,
            num_barbs_x=args.num_barb_x,
            num_barbs_y=args.num_barb_y,
            use_var_for_y=args.var_y,
            subplot_index=subplot_index,
            **args.kwargs,
        )

    if args.xsection_from_1d:
        option_error_check(args, 'field')

        display.plot_time_height_xsection_from_1d_data(
            data_field=args.field,
            pres_field=args.p_field,
            dsname=dsname,
            set_title=args.set_title,
            day_night_background=args.day_night,
            num_time_periods=args.num_time_periods,
            num_y_levels=args.num_y_levels,
            invert_y_axis=args.invert_y_axis,
            subplot_index=subplot_index,
            cbar_label=args.cb_label,
            **args.kwargs,
        )

    if args.time_height_scatter:
        option_error_check(args, 'field')

        display.time_height_scatter(
            data_field=args.field,
            dsname=dsname,
            cmap=args.cmap,
            alt_label=args.alt_label,
            alt_field=args.alt_field,
            cb_label=args.cb_label,
            **args.kwargs,
        )

    if args.qc:
        option_error_check(args, 'field')
        display.qc_flag_block_plot(
            data_field=args.field,
            dsname=dsname,
            subplot_index=subplot_index,
            time_rng=args.time_rng,
            assessment_color=args.assessment_color,
            **args.kwargs,
        )

    if args.fill_between:
        option_error_check(args, 'field')

        display.fill_between(
            field=args.field,
            dsname=dsname,
            subplot_index=subplot_index,
            set_title=args.set_title,
            secondary_y=args.secondary_y,
            **args.kwargs,
        )

    if args.multi_panel:
        option_error_check(args, ['fields', 'plot_type'], check_all=True)

        for i, j, k in zip(args.fields, subplot_index, args.plot_type):
            if k == 'plot':
                display.plot(
                    field=i,
                    dsname=dsname,
                    cmap=args.cmap,
                    set_title=args.set_title,
                    add_nan=args.add_nan,
                    subplot_index=j,
                    use_var_for_y=args.var_y,
                    day_night_background=args.day_night,
                    invert_y_axis=args.invert_y_axis,
                    abs_limits=args.abs_limits,
                    time_rng=args.time_rng,
                    assessment_overplot=args.assessment_overplot,
                    assessment_overplot_category=args.overplot_category,
                    assessment_overplot_category_color=args.category_color,
                    force_line_plot=args.force_line_plot,
                    labels=args.labels,
                    cbar_label=args.cb_label,
                    secondary_y=args.secondary_y,
                    **args.kwargs,
                )

            if k == 'qc':
                display.qc_flag_block_plot(
                    data_field=i,
                    dsname=dsname,
                    subplot_index=j,
                    time_rng=args.time_rng,
                    assessment_color=args.assessment_color,
                    **args.kwargs,
                )

    plt.savefig(args.out_path)
    plt.show()
    plt.close(display.fig)

    ds.close()


def histogram(args):
    drop_vars = find_drop_vars(args)

    ds = act.io.arm.read_arm_netcdf(args.file_path, drop_variables=drop_vars)

    subplot_shape = args.subplot_shape
    subplot_index = args.subplot_index

    dsname = args.dsname
    if dsname == _default_dsname:
        try:
            dsname = ds.attrs['datastream']
        except KeyError:
            pass

    display = act.plotting.DistributionDisplay(
        {dsname: ds}, figsize=args.figsize, subplot_shape=subplot_shape
    )

    if args.stacked_bar_graph:
        display.plot_stacked_bar_graph(
            field=args.field,
            dsname=dsname,
            bins=args.bins,
            density=args.density,
            sortby_field=args.sortby_field,
            sortby_bins=args.sortby_bins,
            set_title=args.set_title,
            subplot_index=subplot_index,
            **args.kwargs,
        )

    if args.size_dist:
        display.plot_size_distribution(
            field=args.field,
            bins=args.bin_field,
            time=args.time,
            dsname=dsname,
            set_title=args.set_title,
            subplot_index=subplot_index,
            **args.kwargs,
        )

    if args.stairstep:
        display.plot_stairstep_graph(
            field=args.field,
            dsname=dsname,
            bins=args.bins,
            density=args.density,
            sortby_field=args.sortby_field,
            sortby_bins=args.sortby_bins,
            set_title=args.set_title,
            subplot_index=subplot_index,
            **args.kwargs,
        )

    if args.heatmap:
        display.plot_heatmap(
            x_field=args.x_field,
            y_field=args.y_field,
            dsname=dsname,
            x_bins=args.x_bins,
            y_bins=args.y_bins,
            set_title=args.set_title,
            density=args.density,
            subplot_index=subplot_index,
            **args.kwargs,
        )

    plt.savefig(args.out_path)
    plt.show()
    plt.close(display.fig)

    ds.close()


def contour(args):
    files = glob.glob(args.file_path)
    files.sort()

    time = args.time
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = act.io.arm.read_arm_netcdf(f)
        data.update({f: ds})
        fields.update({f: args.fields})
        wind_fields.update({f: args.wind_fields})
        station_fields.update({f: args.station_fields})

    display = act.plotting.ContourDisplay(data, figsize=args.figsize)

    if args.create_contour:
        display.create_contour(
            fields=fields,
            time=time,
            function=args.function,
            grid_delta=args.grid_delta,
            grid_buffer=args.grid_buffer,
            subplot_index=args.subplot_index,
            **args.kwargs,
        )

    if args.contourf:
        display.contourf(
            x=args.x, y=args.y, z=args.z, subplot_index=args.subplot_index, **args.kwargs
        )

    if args.plot_contour:
        display.contour(
            x=args.x, y=args.y, z=args.z, subplot_index=args.subplot_index, **args.kwargs
        )

    if args.vectors_spd_dir:
        display.plot_vectors_from_spd_dir(
            fields=wind_fields,
            time=time,
            mesh=args.mesh,
            function=args.function,
            grid_delta=args.grid_delta,
            grid_buffer=args.grid_buffer,
            subplot_index=args.subplot_index,
            **args.kwargs,
        )

    if args.barbs:
        display.barbs(
            x=args.x, y=args.y, u=args.u, v=args.v, subplot_index=args.subplot_index, **args.kwargs
        )

    if args.plot_station:
        display.plot_station(
            fields=station_fields,
            time=time,
            text_color=args.text_color,
            subplot_index=args.subplot_index,
            **args.kwargs,
        )

    plt.savefig(args.out_path)
    plt.show()
    plt.close(display.fig)

    ds.close()


# Define new funciton for argparse to allow specific rules for
# parsing files containing arguments. This works by this function being
# called for each line in the configuration file.
def convert_arg_line_to_args(line):
    for arg in line.split():
        if not arg.strip():  # If empty line or only white space skip
            continue
        if arg.startswith('#'):  # If line starts with comment skip
            break
        yield arg


def main():
    prefix_char = '@'
    parser = argparse.ArgumentParser(
        description=(
            f'Create plot from a data file. Can use command line opitons '
            f'or point to a configuration file using {prefix_char} character.'
        )
    )

    # Allow user to reference a file by using the @ symbol for a specific
    # argument value
    parser = argparse.ArgumentParser(fromfile_prefix_chars=prefix_char)

    # Update the file parsing logic to skip commented lines
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument(
        '-f',
        '--file_path',
        type=str,
        required=True,
        help=(
            'Required: Full path to file for creating Plot. For multiple '
            'files use terminal syntax for matching muliple files. '
            'For example "sgpmetE13.b1.202007*.*.nc" will match all files '
            'for the month of July in 2020. Need to use double quotes '
            'to stop terminal from expanding the search, and let the '
            'python program perform search.'
        ),
    )
    out_path_default = 'image.png'
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        default=out_path_default,
        help=(
            "Full path filename to use for saving image. "
            "Default is '{out_path_default}'. If only a path is given "
            "will use that path with image name '{out_path_default}', "
            "else will use filename given."
        ),
    )
    parser.add_argument('-fd', '--field', type=str, default=None, help='Name of the field to plot')
    parser.add_argument(
        '-fds',
        '--fields',
        nargs='+',
        type=str,
        default=None,
        help='Name of the fields to use to plot',
    )
    parser.add_argument(
        '-wfs',
        '--wind_fields',
        nargs='+',
        type=str,
        default=None,
        help='Wind field names used to plot',
    )
    parser.add_argument(
        '-sfs',
        '--station_fields',
        nargs='+',
        type=str,
        default=None,
        help='Station field names to plot sites',
    )
    default = 'lat'
    parser.add_argument(
        '-lat',
        '--latitude',
        type=str,
        default=default,
        help=f"Name of latitude variable in file. Default is '{default}'",
    )
    default = 'lon'
    parser.add_argument(
        '-lon',
        '--longitude',
        type=str,
        default=default,
        help=f"Name of longitude variable in file. Default is '{default}'",
    )
    parser.add_argument(
        '-xf', '--x_field', type=str, default=None, help='Name of variable to plot on x axis'
    )
    parser.add_argument(
        '-yf', '--y_field', type=str, default=None, help='Name of variable to plot on y axis'
    )
    parser.add_argument('-x', type=np.array, help='x coordinates or grid for z')
    parser.add_argument('-y', type=np.array, help='y coordinates or grid for z')
    parser.add_argument('-z', type=np.array, help='Values over which to contour')
    default = 'u_wind'
    parser.add_argument(
        '-u',
        '--u_wind',
        type=str,
        default=default,
        help=f"File variable name for u_wind wind component. Default is '{default}'",
    )
    default = 'v_wind'
    parser.add_argument(
        '-v',
        '--v_wind',
        type=str,
        default=default,
        help=f"File variable name for v_wind wind compenent. Default is '{default}'",
    )
    default = 'pres'
    parser.add_argument(
        '-pf',
        '--p_field',
        type=str,
        default=default,
        help=f"File variable name for pressure. Default is '{default}'",
    )
    default = 'tdry'
    parser.add_argument(
        '-tf',
        '--t_field',
        type=str,
        default=default,
        help=f"File variable name for temperature. Default is '{default}'",
    )
    default = 'dp'
    parser.add_argument(
        '-tdf',
        '--td_field',
        type=str,
        default=default,
        help=f"File variable name for dewpoint temperature. Default is '{default}'",
    )
    default = 'wspd'
    parser.add_argument(
        '-sf',
        '--spd_field',
        type=str,
        default=default,
        help=f"File variable name for wind speed. Default is '{default}'",
    )
    default = 'deg'
    parser.add_argument(
        '-df',
        '--dir_field',
        type=str,
        default=default,
        help=f"File variable name for wind direction. Default is '{default}'",
    )
    parser.add_argument('-al', '--alt_label', type=str, default=None, help='Altitude axis label')
    default = 'alt'
    parser.add_argument(
        '-af',
        '--alt_field',
        type=str,
        default=default,
        help=f"File variable name for altitude. Default is '{default}'",
    )
    global _default_dsname
    _default_dsname = 'act_datastream'
    parser.add_argument(
        '-ds',
        '--dsname',
        type=str,
        default=_default_dsname,
        help=f"Name of datastream to plot. Default is '{_default_dsname}'",
    )
    default = '(0, )'
    parser.add_argument(
        '-si',
        '--subplot_index',
        type=ast.literal_eval,
        default=default,
        help=f'Index of the subplot via tuple syntax. '
        f'Example for two plots is "(0,), (1,)". '
        f"Default is '{default}'",
    )
    default = (1,)
    parser.add_argument(
        '-ss',
        '--subplot_shape',
        nargs='+',
        type=int,
        default=default,
        help=(
            f'The number of (rows, columns) '
            f'for the subplots in the display. '
            f'Default is {default}'
        ),
    )
    plot_type_options = ['plot', 'qc']
    parser.add_argument(
        '-pt',
        '--plot_type',
        nargs='+',
        type=str,
        help=f'Type of plot to make. Current options include: ' f'{plot_type_options}',
    )
    parser.add_argument(
        '-vy',
        '--var_y',
        type=str,
        default=None,
        help=(
            'Set this to the name of a data variable in '
            'the Dataset to use as the y-axis variable '
            'instead of the default dimension.'
        ),
    )
    parser.add_argument(
        '-plp',
        '--plevels_plot',
        type=np.array,
        default=None,
        help='Pressure levels to plot the wind barbs on.',
    )
    parser.add_argument('-cbl', '--cb_label', type=str, default=None, help='Colorbar label to use')
    parser.add_argument('-st', '--set_title', type=str, default=None, help='Title for the plot')
    default = 0.08
    parser.add_argument(
        '-pb',
        '--plot_buffer',
        type=float,
        default=default,
        help=(
            f'Buffer to add around data on plot in lat ' f'and lon dimension. Default is {default}'
        ),
    )
    default = 'terrain-background'
    parser.add_argument(
        '-sm',
        '--stamen',
        type=str,
        default=default,
        help=f"Dataset to use for background image. Default is '{default}'",
    )
    default = 8
    parser.add_argument(
        '-tl',
        '--tile',
        type=int,
        default=default,
        help=f'Tile zoom to use with background image. Default is {default}',
    )
    parser.add_argument(
        '-cfs',
        '--cfeatures',
        nargs='+',
        type=str,
        default=None,
        help='Cartopy feature to add to plot',
    )
    parser.add_argument(
        '-txt',
        '--text',
        type=json.loads,
        default=None,
        help=(
            'Dictionary of {text:[lon,lat]} to add to plot. '
            'Can have more than one set of text to add.'
        ),
    )
    default = 'rainbow'
    parser.add_argument(
        '-cm', '--cmap', default=default, help=f"colormap to use. Defaut is '{default}'"
    )
    parser.add_argument(
        '-abl',
        '--abs_limits',
        nargs='+',
        type=float,
        default=(None, None),
        help=(
            'Sets the bounds on plot limits even if data '
            'values exceed those limits. Y axis limits. Default is no limits.'
        ),
    )
    parser.add_argument(
        '-tr',
        '--time_rng',
        nargs='+',
        type=float,
        default=None,
        help=('List or tuple with (min,max) values to set the ' 'x-axis range limits'),
    )
    default = 20
    parser.add_argument(
        '-nd',
        '--num_dir',
        type=int,
        default=default,
        help=(f'Number of directions to splot the wind rose into. ' f'Default is {default}'),
    )
    parser.add_argument(
        '-sb',
        '--spd_bins',
        nargs='+',
        type=float,
        default=None,
        help='Bin boundaries to sort the wind speeds into',
    )
    default = 3
    parser.add_argument(
        '-ti',
        '--tick_interval',
        type=int,
        default=default,
        help=(
            f'Interval (in percentage) for the ticks ' f'on the radial axis. Default is {default}'
        ),
    )
    parser.add_argument(
        '-ac',
        '--assessment_color',
        type=json.loads,
        default=None,
        help=('dictionary lookup to override default ' 'assessment to color'),
    )
    default = False
    parser.add_argument(
        '-ao',
        '--assessment_overplot',
        default=default,
        action='store_true',
        help=(
            f'Option to overplot quality control colored '
            f'symbols over plotted data using '
            f'flag_assessment categories. Default is {default}'
        ),
    )
    default = {'Incorrect': ['Bad', 'Incorrect'], 'Suspect': ['Indeterminate', 'Suspect']}
    parser.add_argument(
        '-oc',
        '--overplot_category',
        type=json.loads,
        default=default,
        help=(
            f'Look up to categorize assessments into groups. '
            f'This allows using multiple terms for the same '
            f'quality control level of failure. '
            f'Also allows adding more to the defaults. Default is {default}'
        ),
    )
    default = {'Incorrect': 'red', 'Suspect': 'orange'}
    parser.add_argument(
        '-co',
        '--category_color',
        type=json.loads,
        default=default,
        help=(
            f'Lookup to match overplot category color to '
            f'assessment grouping. Default is {default}'
        ),
    )
    parser.add_argument(
        '-flp',
        '--force_line_plot',
        default=False,
        action='store_true',
        help='Option to plot 2D data as 1D line plots',
    )
    parser.add_argument(
        '-l',
        '--labels',
        nargs='+',
        default=False,
        type=str,
        help=(
            'Option to overwrite the legend labels. '
            'Must have same dimensions as number of '
            'lines plottes.'
        ),
    )
    parser.add_argument(
        '-sy',
        '--secondary_y',
        default=False,
        action='store_true',
        help='Option to plot on secondary y axis',
    )
    if CARTOPY_AVAILABLE:
        default = ccrs.PlateCarree()
        parser.add_argument(
            '-prj',
            '--projection',
            type=str,
            default=default,
            help=f"Projection to use on plot. Default is {default}",
        )
    default = 20
    parser.add_argument(
        '-bx',
        '--num_barb_x',
        type=int,
        default=default,
        help=f'Number of wind barbs to plot in the x axis. Default is {default}',
    )
    default = 20
    parser.add_argument(
        '-by',
        '--num_barb_y',
        type=int,
        default=default,
        help=f"Number of wind barbs to plot in the y axis. Default is {default}",
    )
    default = 20
    parser.add_argument(
        '-tp',
        '--num_time_periods',
        type=int,
        default=default,
        help=f'Set how many time periods. Default is {default}',
    )
    parser.add_argument(
        '-bn', '--bins', nargs='+', type=int, default=None, help='histogram bin boundaries to use'
    )
    parser.add_argument(
        '-bf',
        '--bin_field',
        type=str,
        default=None,
        help=('name of the field that stores the ' 'bins for the spectra'),
    )
    parser.add_argument(
        '-xb',
        '--x_bins',
        nargs='+',
        type=int,
        default=None,
        help='Histogram bin boundaries to use for x axis variable',
    )
    parser.add_argument(
        '-yb',
        '--y_bins',
        nargs='+',
        type=int,
        default=None,
        help='Histogram bin boundaries to use for y axis variable',
    )
    parser.add_argument('-t', '--time', type=str, default=None, help='Time period to be plotted')
    parser.add_argument(
        '-sbf',
        '--sortby_field',
        type=str,
        default=None,
        help='Sort histograms by a given field parameter',
    )
    parser.add_argument(
        '-sbb',
        '--sortby_bins',
        nargs='+',
        type=int,
        default=None,
        help='Bins to sort the histograms by',
    )
    default = 20
    parser.add_argument(
        '-nyl',
        '--num_y_levels',
        type=int,
        default=default,
        help=f'Number of levels in the y axis to use. Default is {default}',
    )
    parser.add_argument(
        '-sk',
        '--sel_kwargs',
        type=json.loads,
        default=None,
        help=('The keyword arguments to pass into ' ':py:func:`xarray.DataArray.sel`'),
    )
    parser.add_argument(
        '-ik',
        '--isel_kwargs',
        type=json.loads,
        default=None,
        help=('The keyword arguments to pass into ' ':py:func:`xarray.DataArray.sel`'),
    )
    default = 'cubic'
    parser.add_argument(
        '-fn',
        '--function',
        type=str,
        default=default,
        help=(
            f'Defaults to cubic function for interpolation. '
            f'See scipy.interpolate.Rbf for additional options. '
            f'Default is {default}'
        ),
    )
    default = 0.1
    parser.add_argument(
        '-gb',
        '--grid_buffer',
        type=float,
        default=default,
        help=f'Buffer to apply to grid. Default is {default}',
    )
    default = (0.01, 0.01)
    parser.add_argument(
        '-gd',
        '--grid_delta',
        nargs='+',
        type=float,
        default=default,
        help=f'X and Y deltas for creating grid. Default is {default}',
    )
    parser.add_argument(
        '-fg',
        '--figsize',
        nargs='+',
        type=float,
        default=None,
        help='Width and height in inches of figure',
    )
    default = 'white'
    parser.add_argument(
        '-tc',
        '--text_color',
        type=str,
        default=default,
        help=f"Color of text. Default is '{default}'",
    )
    parser.add_argument(
        '-kwargs',
        type=json.loads,
        default=dict(),
        help='keyword arguments to use in plotting function',
    )
    parser.add_argument(
        '-pk',
        '--plot_kwargs',
        type=json.loads,
        default=dict(),
        help=("Additional keyword arguments to pass " "into MetPy's SkewT.plot"),
    )
    parser.add_argument(
        '-pbk',
        '--plot_barbs_kwargs',
        type=json.loads,
        default=dict(),
        help=("Additional keyword arguments to pass " "into MetPy's SkewT.plot_barbs"),
    )
    default = True
    parser.add_argument(
        '-cu',
        '--cleanup',
        default=default,
        action='store_false',
        help=f'Turn off standard methods for obj cleanup. Default is {default}',
    )
    parser.add_argument(
        '-gl',
        '--gridlines',
        default=False,
        action='store_true',
        help='Use latitude and lingitude gridlines.',
    )
    parser.add_argument(
        '-cl',
        '--coastlines',
        default=False,
        action='store_true',
        help='Plot coastlines on geographical map',
    )
    parser.add_argument(
        '-bg',
        '--background',
        default=False,
        action='store_true',
        help='Plot a stock image background',
    )
    parser.add_argument(
        '-nan', '--add_nan', default=False, action='store_true', help='Fill in data gaps with NaNs'
    )
    parser.add_argument(
        '-dn',
        '--day_night',
        default=False,
        action='store_true',
        help=("Fill in color coded background according " "to time of day."),
    )
    parser.add_argument(
        '-yr', '--set_yrange', default=None, nargs=2, help=("Set the yrange for the specific plot")
    )
    parser.add_argument(
        '-iya', '--invert_y_axis', default=False, action='store_true', help='Invert y axis'
    )
    parser.add_argument(
        '-sp',
        '--show_parcel',
        default=False,
        action='store_true',
        help='set to true to plot the parcel path.',
    )
    parser.add_argument(
        '-cape',
        '--shade_cape',
        default=False,
        action='store_true',
        help='set to true to shade regions of cape.',
    )
    parser.add_argument(
        '-cin',
        '--shade_cin',
        default=False,
        action='store_true',
        help='set to true to shade regions of cin.',
    )
    parser.add_argument(
        '-d',
        '--density',
        default=False,
        action='store_true',
        help='Plot a p.d.f. instead of a frequency histogram',
    )
    parser.add_argument(
        '-m',
        '--mesh',
        default=False,
        action='store_true',
        help=('Set to True to interpolate u and v to ' 'grid and create wind barbs'),
    )
    parser.add_argument(
        '-uv',
        '--from_u_and_v',
        default=False,
        action='store_true',
        help='Create SkewTPLot with u and v wind',
    )
    parser.add_argument(
        '-sd',
        '--from_spd_and_dir',
        default=False,
        action='store_true',
        help='Create SkewTPlot with wind speed and direction',
    )
    parser.add_argument(
        '-px',
        '--plot_xsection',
        default=False,
        action='store_true',
        help='plots a cross section whose x and y coordinates',
    )
    parser.add_argument(
        '-pxm',
        '--xsection_map',
        default=False,
        action='store_true',
        help='plots a cross section of 2D data on a geographical map',
    )
    parser.add_argument(
        '-p', '--plot', default=False, action='store_true', help='Makes a time series plot'
    )
    parser.add_argument(
        '-mp',
        '--multi_panel',
        default=False,
        action='store_true',
        help='Makes a 2 panel timeseries plot',
    )
    parser.add_argument(
        '-qc',
        '--qc',
        default=False,
        action='store_true',
        help='Create time series plot of embedded quality control values',
    )
    parser.add_argument(
        '-fb',
        '--fill_between',
        default=False,
        action='store_true',
        help='makes a fill betweem plot based on matplotlib',
    )
    parser.add_argument(
        '-bsd',
        '--barbs_spd_dir',
        default=False,
        action='store_true',
        help=('Makes time series plot of wind barbs ' 'using wind speed and dir.'),
    )
    parser.add_argument(
        '-buv',
        '--barbs_u_v',
        default=False,
        action='store_true',
        help=('Makes time series plot of wind barbs ' 'using u and v wind components.'),
    )
    parser.add_argument(
        '-pxs',
        '--xsection_from_1d',
        default=False,
        action='store_true',
        help='Will plot a time-height cross section from 1D dataset',
    )
    parser.add_argument(
        '-ths',
        '--time_height_scatter',
        default=False,
        action='store_true',
        help='Create a scatter time series plot',
    )
    parser.add_argument(
        '-sbg',
        '--stacked_bar_graph',
        default=False,
        action='store_true',
        help='Create stacked bar graph histogram',
    )
    parser.add_argument(
        '-psd',
        '--size_dist',
        default=False,
        action='store_true',
        help='Plots a stairstep plot of size distribution',
    )
    parser.add_argument(
        '-sg',
        '--stairstep',
        default=False,
        action='store_true',
        help='Plots stairstep plot of a histogram',
    )
    parser.add_argument(
        '-hm',
        '--heatmap',
        default=False,
        action='store_true',
        help='Plot a heatmap histogram from 2 variables',
    )
    parser.add_argument(
        '-cc',
        '--create_contour',
        default=False,
        action='store_true',
        help='Extracts, grids, and creates a contour plot',
    )
    parser.add_argument(
        '-cf',
        '--contourf',
        default=False,
        action='store_true',
        help=('Base function for filled contours if user ' 'already has data gridded'),
    )
    parser.add_argument(
        '-ct',
        '--plot_contour',
        default=False,
        action='store_true',
        help=('Base function for contours if user ' 'already has data gridded'),
    )
    parser.add_argument(
        '-vsd',
        '--vectors_spd_dir',
        default=False,
        action='store_true',
        help='Extracts, grids, and creates a contour plot.',
    )
    parser.add_argument(
        '-b', '--barbs', default=False, action='store_true', help='Base function for wind barbs.'
    )
    parser.add_argument(
        '-ps',
        '--plot_station',
        default=False,
        action='store_true',
        help='Extracts, grids, and creates a contour plot',
    )

    # The mutually exclusive but one requried group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-gp',
        '--geodisplay',
        dest='action',
        action='store_const',
        const=geodisplay,
        help='Set to genereate a geographic plot',
    )
    group.add_argument(
        '-skt',
        '--skewt',
        dest='action',
        action='store_const',
        const=skewt,
        help='Set to genereate a skew-t plot',
    )
    group.add_argument(
        '-xs',
        '--xsection',
        dest='action',
        action='store_const',
        const=xsection,
        help='Set to genereate a XSection plot',
    )
    group.add_argument(
        '-wr',
        '--wind_rose',
        dest='action',
        action='store_const',
        const=wind_rose,
        help='Set to genereate a wind rose plot',
    )
    group.add_argument(
        '-ts',
        '--timeseries',
        dest='action',
        action='store_const',
        const=timeseries,
        help='Set to genereate a timeseries plot',
    )
    group.add_argument(
        '-c',
        '--contour',
        dest='action',
        action='store_const',
        const=contour,
        help='Set to genereate a contour plot',
    )
    group.add_argument(
        '-hs',
        '--histogram',
        dest='action',
        action='store_const',
        const=histogram,
        help='Set to genereate a histogram plot',
    )

    args = parser.parse_args()

    # Check if a path but no file name is given. If so use a default name.
    out_path = pathlib.Path(args.out_path)
    if out_path.is_dir():
        args.out_path = str(pathlib.Path(out_path, out_path_default))

    args.action(args)


if __name__ == '__main__':
    main()
