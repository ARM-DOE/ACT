"""
===========================
act.qc (act.qc)
===========================

.. currentmodule:: act.qc

This module contains procedures for working with QC information
and for applying tests to data

.. autosummary::
    :toctree: generated/

    arm.add_dqr_to_qc
    clean.CleanDataset
    qcfilter.QCFilter
    qcfilter.parse_bit
    qcfilter.set_bit
    qcfilter.unset_bit
    qcfilter.QCFilter.add_delta_test
    qcfilter.QCFilter.add_difference_test
    qcfilter.QCFilter.add_equal_to_test
    qcfilter.QCFilter.add_greater_test
    qcfilter.QCFilter.add_greater_equal_test
    qcfilter.QCFilter.add_inside_test
    qcfilter.QCFilter.add_less_test
    qcfilter.QCFilter.add_less_equal_test
    qcfilter.QCFilter.add_missing_value_test
    qcfilter.QCFilter.add_not_equal_to_test
    qcfilter.QCFilter.add_outside_test
    qcfilter.QCFilter.add_persistence_test
    qcfilter.QCFilter.add_test
    qcfilter.QCFilter.available_bit
    qcfilter.QCFilter.check_for_ancillary_qc
    qcfilter.QCFilter.compare_time_series_trends
    qcfilter.QCFilter.create_qc_variable
    qcfilter.QCFilter.get_qc_test_mask
    qcfilter.QCFilter.get_masked_data
    qcfilter.QCFilter.remove_test
    qcfilter.QCFilter.set_test
    qcfilter.QCFilter.unset_test
    qcfilter.QCFilter.update_ancillary_variable
    qctests.QCTests
    radiometer_tests.fft_shading_test
"""

from . import qcfilter
from . import qctests, comparison_tests
from . import clean
from . import arm
from . import radiometer_tests
