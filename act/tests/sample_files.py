"""
act.tests.sample_files
======================

Sample data file for use in testing. These files should only
be used for testing ACT.

-- autosummary::
    :toctree: generated/

    EXAMPLE_SONDE1
    EXAMPLE_LCL1
"""
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

EXAMPLE_MET1 = os.path.join(DATA_PATH, 'sgpmetE13.b1.20190101.000000.cdf')
EXAMPLE_CEIL1 = os.path.join(DATA_PATH, 'sgpceilC1.b1.20190101.000000.nc')
EXAMPLE_SONDE1 = os.path.join(DATA_PATH, 'sgpsondewnpnC1.b1.20190101.053200.cdf')
EXAMPLE_LCL1 = os.path.join(DATA_PATH, 'met_lcl.nc')
EXAMPLE_SONDE_WILDCARD = os.path.join(DATA_PATH, 'sgpsondewnpn*.cdf')
EXAMPLE_MET_WILDCARD = os.path.join(DATA_PATH, 'sgpmet*.cdf')
EXAMPLE_CEIL_WILDCARD = os.path.join(DATA_PATH, 'sgpceil*.cdf')
