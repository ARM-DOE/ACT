import numpy as np

from act.io.arm import read_arm_netcdf
from act.tests import EXAMPLE_MET1
from act.qc.qcfilter import set_bit


def test_qc_summary():

    for cleanup in [False, True]:
        ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=not cleanup)
        for var_name in ['temp_mean', 'rh_mean']:
            qc_var_name = f'qc_{var_name}'
            qc_data = ds[qc_var_name].values

            assert np.sum(qc_data) == 0

            index_4 = np.arange(100, 200)
            qc_data[index_4] = set_bit(qc_data[index_4], 4)
            index_1 = np.arange(170, 230)
            qc_data[index_1] = set_bit(qc_data[index_1], 1)
            index_2 = np.arange(250, 400)
            qc_data[index_2] = set_bit(qc_data[index_2], 2)
            index_3 = np.arange(450, 510)
            qc_data[index_3] = set_bit(qc_data[index_3], 3)
            ds[qc_var_name].values = qc_data

        result = ds.qcfilter.create_qc_summary(cleanup_qc=cleanup)

        assert 'flag_masks' not in result[qc_var_name].attrs.keys()
        assert isinstance(result[qc_var_name].attrs['flag_values'], list)

        assert np.sum(result[qc_var_name].values) == 610

        qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Indeterminate')
        assert np.all(np.where(qc_ma.mask)[0] == np.arange(100, 170))

        qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
        index = np.concatenate([index_1, index_2, index_3])
        assert np.all(np.where(qc_ma.mask)[0] == index)

        assert "Quality control summary implemented by ACT" in result.attrs['history']


def test_qc_summary_multiple_assessment_names():

    ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=True)
    var_name = 'temp_mean'
    qc_var_name = f'qc_{var_name}'
    qc_data = ds[qc_var_name].values

    assert np.sum(qc_data) == 0

    index_4 = np.arange(200, 300)
    qc_data[index_4] = set_bit(qc_data[index_4], 4)
    index_1 = np.arange(270, 330)
    qc_data[index_1] = set_bit(qc_data[index_1], 1)
    index_2 = np.arange(350, 500)
    qc_data[index_2] = set_bit(qc_data[index_2], 2)
    index_3 = np.arange(550, 610)
    qc_data[index_3] = set_bit(qc_data[index_3], 3)
    ds[qc_var_name].values = qc_data

    index_5 = np.arange(50, 150)
    ds.qcfilter.add_test(
        var_name,
        index=index_5,
        test_meaning='Testing Suspect',
        test_assessment='Suspect')

    index_6 = np.arange(130, 210)
    ds.qcfilter.add_test(
        var_name,
        index=index_6,
        test_meaning='Testing Incorrect',
        test_assessment='Incorrect')

    result = ds.qcfilter.create_qc_summary()

    assert result[qc_var_name].attrs['flag_assessments'] ==\
        ['Passing', 'Suspect', 'Indeterminate', 'Incorrect', 'Bad']

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Indeterminate')
    assert np.sum(np.where(qc_ma.mask)[0]) == 14370

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Suspect')
    assert np.sum(np.where(qc_ma.mask)[0]) == 7160

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Bad')
    assert np.sum(np.where(qc_ma.mask)[0]) == 116415

    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Incorrect')
    assert np.sum(np.where(qc_ma.mask)[0]) == 13560

    assert np.sum(np.where(result[qc_var_name].values == 0)) == 884575
    qc_ma = result.qcfilter.get_masked_data(var_name, rm_assessments='Passing')
    assert np.sum(np.where(qc_ma.mask)[0]) == 884575
