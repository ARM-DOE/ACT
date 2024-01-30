import copy

import dask.array as da
import numpy as np
import xarray as xr

from act.io.arm import read_arm_netcdf
from act.qc.bsrn_tests import _calculate_solar_parameters
from act.tests import EXAMPLE_BRS


def test_bsrn_limits_test():
    for use_dask in [False, True]:
        ds = read_arm_netcdf(EXAMPLE_BRS)
        var_names = list(ds.data_vars)
        # Remove QC variables to make testing easier
        for var_name in var_names:
            if var_name.startswith('qc_'):
                del ds[var_name]

        # Add atmospheric temperature fake data
        ds['temp_mean'] = xr.DataArray(
            data=np.full(ds.time.size, 13.5),
            dims=['time'],
            attrs={'long_name': 'Atmospheric air temperature', 'units': 'degC'},
        )

        # Make a short direct variable since BRS does not have one
        ds['short_direct'] = copy.deepcopy(ds['short_direct_normal'])
        ds['short_direct'].attrs['ancillary_variables'] = 'qc_short_direct'
        ds['short_direct'].attrs['long_name'] = 'Shortwave direct irradiance, pyrheliometer'
        _, _ = _calculate_solar_parameters(ds, 'lat', 'lon', 1360.8)
        ds['short_direct'].data = ds['short_direct'].data * 0.5

        # Make up long variable since BRS does not have values
        ds['up_long_hemisp'].data = copy.deepcopy(ds['down_long_hemisp_shaded'].data)
        data = copy.deepcopy(ds['down_short_hemisp'].data)
        ds['up_short_hemisp'].data = data

        # Test that nothing happens when no variable names are provided
        ds.qcfilter.bsrn_limits_test()

        # Mess with data to get tests to trip
        data = ds['down_short_hemisp'].values
        data[200:300] -= 10
        data[800:850] += 330
        data[1340:1380] += 600
        ds['down_short_hemisp'].data = da.from_array(data)

        data = ds['down_short_diffuse_hemisp'].values
        data[200:250] = data[200:250] - 1.9
        data[250:300] = data[250:300] - 3.9
        data[800:850] += 330
        data[1340:1380] += 600
        ds['down_short_diffuse_hemisp'].data = da.from_array(data)

        data = ds['short_direct_normal'].values
        data[200:250] = data[200:250] - 1.9
        data[250:300] = data[250:300] - 3.9
        data[800:850] += 600
        data[1340:1380] += 800
        ds['short_direct_normal'].data = da.from_array(data)

        data = ds['short_direct'].values
        data[200:250] = data[200:250] - 1.9
        data[250:300] = data[250:300] - 3.9
        data[800:850] += 300
        data[1340:1380] += 800
        ds['short_direct'].data = da.from_array(data)

        data = ds['down_long_hemisp_shaded'].values
        data[200:250] = data[200:250] - 355
        data[250:300] = data[250:300] - 400
        data[800:850] += 200
        data[1340:1380] += 400
        ds['down_long_hemisp_shaded'].data = da.from_array(data)

        data = ds['up_long_hemisp'].values
        data[200:250] = data[200:250] - 355
        data[250:300] = data[250:300] - 400
        data[800:850] += 300
        data[1340:1380] += 500
        ds['up_long_hemisp'].data = da.from_array(data)

        ds.qcfilter.bsrn_limits_test(
            gbl_SW_dn_name='down_short_hemisp',
            glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
            direct_normal_SW_dn_name='short_direct_normal',
            glb_SW_up_name='up_short_hemisp',
            glb_LW_dn_name='down_long_hemisp_shaded',
            glb_LW_up_name='up_long_hemisp',
            direct_SW_dn_name='short_direct',
            use_dask=use_dask,
        )

        assert ds['qc_down_short_hemisp'].attrs['flag_masks'] == [1, 2]
        assert (
            ds['qc_down_short_hemisp'].attrs['flag_meanings'][-2]
            == 'Value less than BSRN physically possible limit of -4.0 W/m^2'
        )
        assert (
            ds['qc_down_short_hemisp'].attrs['flag_meanings'][-1]
            == 'Value greater than BSRN physically possible limit'
        )

        assert ds['qc_down_short_diffuse_hemisp'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_down_short_diffuse_hemisp'].attrs['flag_assessments'] == ['Bad', 'Bad']

        assert ds['qc_short_direct'].attrs['flag_masks'] == [1, 2]
        assert ds['qc_short_direct'].attrs['flag_assessments'] == ['Bad', 'Bad']
        assert ds['qc_short_direct'].attrs['flag_meanings'] == [
            'Value less than BSRN physically possible limit of -4.0 W/m^2',
            'Value greater than BSRN physically possible limit',
        ]

        assert ds['qc_short_direct_normal'].attrs['flag_masks'] == [1, 2]
        assert (
            ds['qc_short_direct_normal'].attrs['flag_meanings'][-1]
            == 'Value greater than BSRN physically possible limit'
        )

        assert ds['qc_down_short_hemisp'].attrs['flag_masks'] == [1, 2]
        assert (
            ds['qc_down_short_hemisp'].attrs['flag_meanings'][-1]
            == 'Value greater than BSRN physically possible limit'
        )

        assert ds['qc_up_short_hemisp'].attrs['flag_masks'] == [1, 2]
        assert (
            ds['qc_up_short_hemisp'].attrs['flag_meanings'][-1]
            == 'Value greater than BSRN physically possible limit'
        )

        assert ds['qc_up_long_hemisp'].attrs['flag_masks'] == [1, 2]
        assert (
            ds['qc_up_long_hemisp'].attrs['flag_meanings'][-1]
            == 'Value greater than BSRN physically possible limit of 900.0 W/m^2'
        )

        ds.qcfilter.bsrn_limits_test(
            test='Extremely Rare',
            gbl_SW_dn_name='down_short_hemisp',
            glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
            direct_normal_SW_dn_name='short_direct_normal',
            glb_SW_up_name='up_short_hemisp',
            glb_LW_dn_name='down_long_hemisp_shaded',
            glb_LW_up_name='up_long_hemisp',
            direct_SW_dn_name='short_direct',
            use_dask=use_dask,
        )

        assert ds['qc_down_short_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_down_short_diffuse_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_short_direct'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_short_direct_normal'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_up_short_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]
        assert ds['qc_up_long_hemisp'].attrs['flag_masks'] == [1, 2, 4, 8]

        assert (
            ds['qc_up_long_hemisp'].attrs['flag_meanings'][-1]
            == 'Value greater than BSRN extremely rare limit of 700.0 W/m^2'
        )

        assert (
            ds['qc_down_long_hemisp_shaded'].attrs['flag_meanings'][-1]
            == 'Value greater than BSRN extremely rare limit of 500.0 W/m^2'
        )

        # down_short_hemisp
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=1)
        assert np.sum(result) == 100
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=2)
        assert np.sum(result) == 26
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=3)
        assert np.sum(result) == 337
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=4)
        assert np.sum(result) == 66

        # down_short_diffuse_hemisp
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=1)
        assert np.sum(result) == 50
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=2)
        assert np.sum(result) == 56
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=3)
        assert np.sum(result) == 100
        result = ds.qcfilter.get_qc_test_mask('down_short_diffuse_hemisp', test_number=4)
        assert np.sum(result) == 90

        # short_direct_normal
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=1)
        assert np.sum(result) == 46
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=2)
        assert np.sum(result) == 26
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=3)
        assert np.sum(result) == 94
        result = ds.qcfilter.get_qc_test_mask('short_direct_normal', test_number=4)
        assert np.sum(result) == 38

        # short_direct_normal
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=1)
        assert np.sum(result) == 41
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=2)
        assert np.sum(result) == 607
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=3)
        assert np.sum(result) == 89
        result = ds.qcfilter.get_qc_test_mask('short_direct', test_number=4)
        assert np.sum(result) == 79

        # down_long_hemisp_shaded
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=1)
        assert np.sum(result) == 50
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=2)
        assert np.sum(result) == 40
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=3)
        assert np.sum(result) == 89
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=4)
        assert np.sum(result) == 90

        # up_long_hemisp
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=1)
        assert np.sum(result) == 50
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=2)
        assert np.sum(result) == 40
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=3)
        assert np.sum(result) == 89
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=4)
        assert np.sum(result) == 90

        # Change data values to trip tests
        ds['down_short_diffuse_hemisp'].values[0:100] = (
            ds['down_short_diffuse_hemisp'].values[0:100] + 100
        )
        ds['up_long_hemisp'].values[0:100] = ds['up_long_hemisp'].values[0:100] - 200

        ds.qcfilter.bsrn_comparison_tests(
            [
                'Global over Sum SW Ratio',
                'Diffuse Ratio',
                'SW up',
                'LW down to air temp',
                'LW up to air temp',
                'LW down to LW up',
            ],
            gbl_SW_dn_name='down_short_hemisp',
            glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
            direct_normal_SW_dn_name='short_direct_normal',
            glb_SW_up_name='up_short_hemisp',
            glb_LW_dn_name='down_long_hemisp_shaded',
            glb_LW_up_name='up_long_hemisp',
            air_temp_name='temp_mean',
            test_assessment='Indeterminate',
            lat_name='lat',
            lon_name='lon',
            use_dask=use_dask,
        )

        # Ratio of Global over Sum SW
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=5)
        assert np.sum(result) == 190

        # Diffuse Ratio
        result = ds.qcfilter.get_qc_test_mask('down_short_hemisp', test_number=6)
        assert np.sum(result) == 47

        # Shortwave up comparison
        result = ds.qcfilter.get_qc_test_mask('up_short_hemisp', test_number=5)
        assert np.sum(result) == 226

        # Longwave up to air temperature comparison
        result = ds.qcfilter.get_qc_test_mask('up_long_hemisp', test_number=5)
        assert np.sum(result) == 290

        # Longwave down to air temperature compaison
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=5)
        assert np.sum(result) == 976

        # Lonwave down to longwave up comparison
        result = ds.qcfilter.get_qc_test_mask('down_long_hemisp_shaded', test_number=6)
        assert np.sum(result) == 100
