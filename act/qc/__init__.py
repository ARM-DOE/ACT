"""
===========================
act.qc (act.qc)
===========================

.. currentmodule:: act.qc

This module contains procedures for working with QC information

.. autosummary::
    :toctree: generated/

    qcfilter.QCFilter
    qctests.QCTests
    qcfilter.parse_bit
    qcfilter.set_bit
    qcfilter.unset_bit
    qcfilter.QCFilter.get_masked_data
    qcfilter.QCFilter.check_for_ancillary_qc
    qcfilter.QCFilter.create_qc_variable
    qcfilter.QCFilter.add_test
    qcfilter.QCFilter.remove_test
    qcfilter.QCFilter.set_test
    qcfilter.QCFilter.unset_test
    qcfilter.QCFilter.available_bit
    qcfilter.QCFilter.get_qc_test_mask
    qcfilter.QCFilter.update_ancillary_variable
    qcfilter.QCFilter.add_missing_value_test
    qcfilter.QCFilter.add_greater_test
    qcfilter.QCFilter.add_greater_equal_test
    qcfilter.QCFilter.add_less_test
    qcfilter.QCFilter.add_less_equal_test
    qcfilter.QCFilter.add_persistence_test
    qcfilter.QCFilter.add_difference_test
    qcfilter.QCFilter.add_equal_to_test
    qcfilter.QCFilter.add_inside_test
    qcfilter.QCFilter.add_outside_test
    qcfilter.QCFilter.add_not_equal_to_test
"""

from . import qcfilter
from . import qctests
