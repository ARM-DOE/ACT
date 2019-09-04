"""
===============
act.io (act.io)
===============

.. currentmodule:: act.io

This module contains procedures for reading and writing various ARM datasets.

.. autosummary::
    :toctree: generated/

    armfiles.read_netcdf
    armfiles.check_arm_standards
    armfiles.ARMStandardsFlag
    dataset.ACTAccessor
    csvfiles.read_csv
"""

from . import armfiles
from . import dataset
from . import clean
from . import csvfiles
