"""
This module contains sample files used for testing the ARM Community Toolkit.
Files in this module should only be used for testing, not production.

"""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['sample_files'],
    submod_attrs={
        'sample_files': [
            'EXAMPLE_AERI',
            'EXAMPLE_AAF_ICARTT',
            'EXAMPLE_ANL_CSV',
            'EXAMPLE_AOSMET',
            'EXAMPLE_BRS',
            'EXAMPLE_CEIL1',
            'EXAMPLE_CEIL_WILDCARD',
            'EXAMPLE_CO2FLX4M',
            'EXAMPLE_DLPPI',
            'EXAMPLE_EBBR1',
            'EXAMPLE_EBBR2',
            'EXAMPLE_IRTSST',
            'EXAMPLE_LCL1',
            'EXAMPLE_MET1',
            'EXAMPLE_MET_CONTOUR',
            'EXAMPLE_MET_CSV',
            'EXAMPLE_MET_TEST1',
            'EXAMPLE_MET_TEST2',
            'EXAMPLE_MET_WILDCARD',
            'EXAMPLE_METE40',
            'EXAMPLE_MFRSR',
            'EXAMPLE_MMCR',
            'EXAMPLE_MPL_1SAMPLE',
            'EXAMPLE_NAV',
            'EXAMPLE_NOAA_PSL',
            'EXAMPLE_NOAA_PSL_TEMPERATURE',
            'EXAMPLE_RL1',
            'EXAMPLE_SIGMA_MPLV5',
            'EXAMPLE_SIRS',
            'EXAMPLE_SONDE1',
            'EXAMPLE_SONDE_WILDCARD',
            'EXAMPLE_STAMP_WILDCARD',
            'EXAMPLE_SURFSPECALB1MLAWER',
            'EXAMPLE_TWP_SONDE_20060121',
            'EXAMPLE_IRT25m20s',
            'EXAMPLE_HK',
            'EXAMPLE_INI',
            'EXAMPLE_SP2B',
            'EXAMPLE_MET_YAML',
            'EXAMPLE_CLOUDPHASE'
        ]
    },
)
