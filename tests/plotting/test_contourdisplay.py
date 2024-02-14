import matplotlib
import pytest

import act
from act.plotting import ContourDisplay
from act.tests import sample_files

matplotlib.use('Agg')


@pytest.mark.mpl_image_compare(tolerance=10)
def test_contour():
    files = sample_files.EXAMPLE_MET_CONTOUR
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = act.io.arm.read_arm_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contour', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=True, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='red')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_contour_stamp():
    files = sample_files.EXAMPLE_STAMP_WILDCARD
    test = {}
    stamp_fields = {}
    time = '2020-01-01T00:00:00.000000000'
    for f in files:
        ds = f.split('/')[-1]
        nc_ds = act.io.arm.read_arm_netcdf(f)
        test.update({ds: nc_ds})
        stamp_fields.update({ds: ['lon', 'lat', 'plant_water_availability_east']})
        nc_ds.close()

    display = act.plotting.ContourDisplay(test, figsize=(8, 8))
    display.create_contour(fields=stamp_fields, time=time, levels=50, alpha=0.5, twod_dim_value=5)

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_contour2():
    files = sample_files.EXAMPLE_MET_CONTOUR
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = act.io.arm.read_arm_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update({f: ['lon', 'lat', 'atmos_pressure']})

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contour', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=False, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='pink')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_contourf():
    files = sample_files.EXAMPLE_MET_CONTOUR
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = act.io.arm.read_arm_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update(
            {
                f: [
                    'lon',
                    'lat',
                    'atmos_pressure',
                    'temp_mean',
                    'rh_mean',
                    'vapor_pressure_mean',
                    'temp_std',
                ]
            }
        )

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contourf', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=True, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='red')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_contourf2():
    files = sample_files.EXAMPLE_MET_CONTOUR
    time = '2019-05-08T04:00:00.000000000'
    data = {}
    fields = {}
    wind_fields = {}
    station_fields = {}
    for f in files:
        ds = act.io.arm.read_arm_netcdf(f)
        data.update({f: ds})
        fields.update({f: ['lon', 'lat', 'temp_mean']})
        wind_fields.update({f: ['lon', 'lat', 'wspd_vec_mean', 'wdir_vec_mean']})
        station_fields.update(
            {
                f: [
                    'lon',
                    'lat',
                    'atmos_pressure',
                    'temp_mean',
                    'rh_mean',
                    'vapor_pressure_mean',
                    'temp_std',
                ]
            }
        )

    display = ContourDisplay(data, figsize=(8, 8))
    display.create_contour(fields=fields, time=time, levels=50, contour='contourf', cmap='viridis')
    display.plot_vectors_from_spd_dir(
        fields=wind_fields, time=time, mesh=False, grid_delta=(0.1, 0.1)
    )
    display.plot_station(fields=station_fields, time=time, markersize=7, color='pink')

    try:
        return display.fig
    finally:
        matplotlib.pyplot.close(display.fig)
