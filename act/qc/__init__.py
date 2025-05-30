"""
This module contains procedures for working with QC information
and for applying tests to data.

"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        'add_supplemental_qc',
        'arm',
        'bsrn_tests',
        'clean',
        'comparison_tests',
        'qcfilter',
        'qc_summary',
        'qctests',
        'radiometer_tests',
        'sp2',
    ],
    submod_attrs={
        'arm': ['add_dqr_to_qc', 'print_dqr'],
        'qcfilter': [
            'QCFilter',
            'parse_bit',
            'set_bit',
            'unset_bit',
        ],
        'qc_summary': ['QCSummary'],
        'qctests': [
            'QCTests',
        ],
        'clean': ['CleanDataset'],
        'radiometer_tests': [
            'fft_shading_test',
            'fft_shading_test_process',
        ],
        'bsrn_tests': ['QCTests'],
        'comparison_tests': ['QCTests'],
        'add_supplemental_qc': ['read_yaml_supplemental_qc'],
        'sp2': ['SP2ParticleCriteria', 'get_waveform_statistics'],
    },
)
