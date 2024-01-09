import tempfile
from pathlib import Path

import act


def test_calculate_dqr_times():
    ebbr1_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_EBBR1)
    ebbr2_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_EBBR2)
    brs_ds = act.io.arm.read_arm_netcdf(act.tests.sample_files.EXAMPLE_BRS)
    ebbr1_result = act.utils.calculate_dqr_times(ebbr1_ds, variable=['soil_temp_1'], threshold=2)
    ebbr2_result = act.utils.calculate_dqr_times(
        ebbr2_ds, variable=['rh_bottom_fraction'], qc_bit=3, threshold=2
    )
    ebbr3_result = act.utils.calculate_dqr_times(
        ebbr2_ds, variable=['rh_bottom_fraction'], qc_bit=3
    )
    brs_result = act.utils.calculate_dqr_times(
        brs_ds, variable='down_short_hemisp_min', qc_bit=2, threshold=30
    )
    assert ebbr1_result == [('2019-11-25 02:00:00', '2019-11-25 04:30:00')]
    assert ebbr2_result == [('2019-11-30 00:00:00', '2019-11-30 11:00:00')]
    assert brs_result == [('2019-07-05 01:57:00', '2019-07-05 11:07:00')]
    assert ebbr3_result is None
    with tempfile.TemporaryDirectory() as tmpdirname:
        write_file = Path(tmpdirname)
        brs_result = act.utils.calculate_dqr_times(
            brs_ds,
            variable='down_short_hemisp_min',
            qc_bit=2,
            threshold=30,
            txt_path=str(write_file),
        )

    brs_result = act.utils.calculate_dqr_times(
        brs_ds,
        variable='down_short_hemisp_min',
        qc_bit=2,
        threshold=30,
        return_missing=False,
    )
    assert len(brs_result[0]) == 2

    ebbr1_ds.close()
    ebbr2_ds.close()
    brs_ds.close()
