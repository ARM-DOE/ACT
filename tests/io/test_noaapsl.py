import numpy as np
import pytest

import act
from act.io import read_psl_surface_met, read_psl_wind_profiler_temperature
from act.tests import sample_files


def test_read_psl_wind_profiler():
    test_ds_low, test_ds_hi = act.io.noaapsl.read_psl_wind_profiler(
        act.tests.EXAMPLE_NOAA_PSL, transpose=False
    )
    # test dimensions
    assert 'time' and 'HT' in test_ds_low.dims.keys()
    assert 'time' and 'HT' in test_ds_hi.dims.keys()
    assert test_ds_low.dims['time'] == 4
    assert test_ds_hi.dims['time'] == 4
    assert test_ds_low.dims['HT'] == 49
    assert test_ds_hi.dims['HT'] == 50

    # test coordinates
    assert (test_ds_low.coords['HT'][0:5] == np.array([0.151, 0.254, 0.356, 0.458, 0.561])).all()
    assert (
        test_ds_low.coords['time'][0:2]
        == np.array(
            ['2021-05-05T15:00:01.000000000', '2021-05-05T15:15:49.000000000'],
            dtype='datetime64[ns]',
        )
    ).all()

    # test attributes
    assert test_ds_low.attrs['site_identifier'] == 'CTD'
    assert test_ds_low.attrs['data_type'] == 'WINDS'
    assert test_ds_low.attrs['revision_number'] == '5.1'
    assert test_ds_low.attrs['latitude'] == 34.66
    assert test_ds_low.attrs['longitude'] == -87.35
    assert test_ds_low.attrs['elevation'] == 187.0
    assert (
        test_ds_low.attrs['beam_azimuth'] == np.array([38.0, 38.0, 308.0], dtype='float32')
    ).all()
    assert (
        test_ds_low.attrs['beam_elevation'] == np.array([90.0, 74.7, 74.7], dtype='float32')
    ).all()
    assert test_ds_low.attrs['consensus_average_time'] == 24
    assert test_ds_low.attrs['oblique-beam_vertical_correction'] == 0
    assert test_ds_low.attrs['number_of_beams'] == 3
    assert test_ds_low.attrs['number_of_range_gates'] == 49
    assert test_ds_low.attrs['number_of_gates_oblique'] == 49
    assert test_ds_low.attrs['number_of_gates_vertical'] == 49
    assert test_ds_low.attrs['number_spectral_averages_oblique'] == 50
    assert test_ds_low.attrs['number_spectral_averages_vertical'] == 50
    assert test_ds_low.attrs['pulse_width_oblique'] == 708
    assert test_ds_low.attrs['pulse_width_vertical'] == 708
    assert test_ds_low.attrs['inner_pulse_period_oblique'] == 50
    assert test_ds_low.attrs['inner_pulse_period_vertical'] == 50
    assert test_ds_low.attrs['full_scale_doppler_value_oblique'] == 20.9
    assert test_ds_low.attrs['full_scale_doppler_value_vertical'] == 20.9
    assert test_ds_low.attrs['delay_to_first_gate_oblique'] == 4000
    assert test_ds_low.attrs['delay_to_first_gate_vertical'] == 4000
    assert test_ds_low.attrs['spacing_of_gates_oblique'] == 708
    assert test_ds_low.attrs['spacing_of_gates_vertical'] == 708

    # test fields
    assert test_ds_low['RAD1'].shape == (4, 49)
    assert test_ds_hi['RAD1'].shape == (4, 50)
    assert (test_ds_low['RAD1'][0, 0:5] == np.array([0.2, 0.1, 0.1, 0.0, -0.1])).all()
    assert (test_ds_hi['RAD1'][0, 0:5] == np.array([0.1, 0.1, -0.1, 0.0, -0.2])).all()

    assert test_ds_low['SPD'].shape == (4, 49)
    assert test_ds_hi['SPD'].shape == (4, 50)
    assert (test_ds_low['SPD'][0, 0:5] == np.array([2.5, 3.3, 4.3, 4.3, 4.8])).all()
    assert (test_ds_hi['SPD'][0, 0:5] == np.array([3.7, 4.6, 6.3, 5.2, 6.8])).all()

    # test transpose
    test_ds_low, test_ds_hi = act.io.noaapsl.read_psl_wind_profiler(
        act.tests.EXAMPLE_NOAA_PSL, transpose=True
    )
    assert test_ds_low['RAD1'].shape == (49, 4)
    assert test_ds_hi['RAD1'].shape == (50, 4)
    assert test_ds_low['SPD'].shape == (49, 4)
    assert test_ds_hi['SPD'].shape == (50, 4)
    test_ds_low.close()


def test_read_psl_wind_profiler_temperature():
    ds = read_psl_wind_profiler_temperature(act.tests.EXAMPLE_NOAA_PSL_TEMPERATURE)

    assert ds.attrs['site_identifier'] == 'CTD'
    assert ds.attrs['elevation'] == 600.0
    assert ds.T.values[0] == 33.2


def test_read_psl_surface_met():
    ds = read_psl_surface_met(sample_files.EXAMPLE_NOAA_PSL_SURFACEMET)
    assert ds.time.size == 2
    assert np.isclose(np.sum(ds['Pressure'].values), 1446.9)
    assert np.isclose(ds['lat'].values, 38.972425)
    assert ds['lat'].attrs['units'] == 'degree_N'
    assert ds['Upward_Longwave_Irradiance'].attrs['long_name'] == 'Upward Longwave Irradiance'
    assert ds['Upward_Longwave_Irradiance'].dtype.str == '<f4'

    with pytest.raises(Exception):
        ds = read_psl_surface_met('aaa22001.00m')


def test_read_psl_parsivel():
    url = [
        'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200200_stats.txt',
        'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200201_stats.txt',
        'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200202_stats.txt',
    ]

    ds = act.io.noaapsl.read_psl_parsivel(url)
    assert 'number_density_drops' in ds
    assert np.max(ds['number_density_drops'].values) == 355
    assert ds['number_density_drops'].values[10, 10] == 201

    ds = act.io.noaapsl.read_psl_parsivel(
        'https://downloads.psl.noaa.gov/psd2/data/realtime/DisdrometerParsivel/Stats/ctd/2022/002/ctd2200201_stats.txt'
    )
    assert 'number_density_drops' in ds


def test_read_psl_fmcw_moment():
    result = act.discovery.download_noaa_psl_data(
        site='kps', instrument='Radar FMCW Moment', startdate='20220815', hour='06'
    )
    ds = act.io.noaapsl.read_psl_radar_fmcw_moment([result[-1]])
    assert 'range' in ds
    np.testing.assert_almost_equal(ds['reflectivity_uncalibrated'].mean(), 2.37, decimal=2)
    assert ds['range'].max() == 10040.0
    assert len(ds['time'].values) == 115


def test_read_psl_sband_moment():
    result = act.discovery.download_noaa_psl_data(
        site='ctd', instrument='Radar S-band Moment', startdate='20211225', hour='06'
    )
    ds = act.io.noaapsl.read_psl_radar_sband_moment([result[-1]])
    assert 'range' in ds
    np.testing.assert_almost_equal(ds['reflectivity_uncalibrated'].mean(), 1.00, decimal=2)
    assert ds['range'].max() == 9997.0
    assert len(ds['time'].values) == 37
