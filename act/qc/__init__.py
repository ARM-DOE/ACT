"""
This module contains procedures for working with QC information
and for applying tests to data.

"""

import lazy_loader as lazy

# We need to import clean first to register the accessor
from .clean import *
from .qcfilter import QCFilter
from .qctests import QCTests

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        'add_supplemental_qc',
        'arm',
        'bsrn_tests',
        'comparison_tests',
        'qcfilter',
        'qctests',
        'radiometer_tests',
        'sp2',
    ],
    submod_attrs={
        'arm': ['add_dqr_to_qc'],
        'qcfilter': ['QCFilter'],
        'qctests': ['QCTests'],
        'radiometer_tests': ['fft_shading_test'],
        'bsrn_tests': ['QCTests'],
        'comparison_tests': ['QCTests'],
        'add_supplemental_qc': ['read_yaml_supplemental_qc'],
        'sp2': ['SP2ParticleCriteria', 'get_waveform_statistics'],
    },
)
