import copy

import numpy as np

from act.io.arm import read_arm_netcdf
from act.tests import EXAMPLE_MET1


def test_compare_time_series_trends():
    drop_vars = [
        'base_time',
        'time_offset',
        'atmos_pressure',
        'qc_atmos_pressure',
        'temp_std',
        'rh_mean',
        'qc_rh_mean',
        'rh_std',
        'vapor_pressure_mean',
        'qc_vapor_pressure_mean',
        'vapor_pressure_std',
        'wspd_arith_mean',
        'qc_wspd_arith_mean',
        'wspd_vec_mean',
        'qc_wspd_vec_mean',
        'wdir_vec_mean',
        'qc_wdir_vec_mean',
        'wdir_vec_std',
        'tbrg_precip_total',
        'qc_tbrg_precip_total',
        'tbrg_precip_total_corr',
        'qc_tbrg_precip_total_corr',
        'org_precip_rate_mean',
        'qc_org_precip_rate_mean',
        'pwd_err_code',
        'pwd_mean_vis_1min',
        'qc_pwd_mean_vis_1min',
        'pwd_mean_vis_10min',
        'qc_pwd_mean_vis_10min',
        'pwd_pw_code_inst',
        'qc_pwd_pw_code_inst',
        'pwd_pw_code_15min',
        'qc_pwd_pw_code_15min',
        'pwd_pw_code_1hr',
        'qc_pwd_pw_code_1hr',
        'pwd_precip_rate_mean_1min',
        'qc_pwd_precip_rate_mean_1min',
        'pwd_cumul_rain',
        'qc_pwd_cumul_rain',
        'pwd_cumul_snow',
        'qc_pwd_cumul_snow',
        'logger_volt',
        'qc_logger_volt',
        'logger_temp',
        'qc_logger_temp',
        'lat',
        'lon',
        'alt',
    ]
    ds = read_arm_netcdf(EXAMPLE_MET1, drop_variables=drop_vars)
    ds.clean.cleanup()
    ds2 = copy.deepcopy(ds)

    var_name = 'temp_mean'
    qc_var_name = ds.qcfilter.check_for_ancillary_qc(
        var_name, add_if_missing=False, cleanup=False, flag_type=False
    )
    ds.qcfilter.compare_time_series_trends(
        var_name=var_name,
        time_shift=60,
        comp_var_name=var_name,
        comp_dataset=ds2,
        time_qc_threshold=60 * 10,
    )

    test_description = (
        'Time shift detected with Minimum Difference test. Comparison of '
        'temp_mean with temp_mean off by 0 seconds exceeding absolute '
        'threshold of 600 seconds.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_description

    time = ds2['time'].values + np.timedelta64(1, 'h')
    time_attrs = ds2['time'].attrs
    ds2 = ds2.assign_coords({'time': time})
    ds2['time'].attrs = time_attrs

    ds.qcfilter.compare_time_series_trends(
        var_name=var_name, comp_dataset=ds2, time_step=60, time_match_threshhold=50
    )

    test_description = (
        'Time shift detected with Minimum Difference test. Comparison of '
        'temp_mean with temp_mean off by 3600 seconds exceeding absolute '
        'threshold of 900 seconds.'
    )
    assert ds[qc_var_name].attrs['flag_meanings'][-1] == test_description
