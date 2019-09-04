"""
act.tests.sample_files
======================

Sample data file for use in testing. These files should only
be used for testing ACT.

-- autosummary::
    :toctree: generated/

    EXAMPLE_MET1
    EXAMPLE_CEIL1
    EXAMPLE_SONDE1
    EXAMPLE_LCL1
    EXAMPLE_SONDE_WILDCARD
    EXAMPLE_MET_WILDCARD
    EXAMPLE_CEIL_WILDCARD
    EXAMPLE_TWP_SONDE_WILDCARD
    EXAMPLE_ANL_CSV
    EXAMPLE_VISST
"""
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

EXAMPLE_MET1 = os.path.join(DATA_PATH, 'sgpmetE13.b1.20190101.000000.cdf')
EXAMPLE_CEIL1 = os.path.join(DATA_PATH, 'sgpceilC1.b1.20190101.000000.nc')
EXAMPLE_SONDE1 = os.path.join(DATA_PATH,
                              'sgpsondewnpnC1.b1.20190101.053200.cdf')
EXAMPLE_LCL1 = os.path.join(DATA_PATH, 'met_lcl.nc')
EXAMPLE_SONDE_WILDCARD = os.path.join(DATA_PATH, 'sgpsondewnpn*.cdf')
EXAMPLE_MET_WILDCARD = os.path.join(DATA_PATH, 'sgpmet*201901*.cdf')
EXAMPLE_MET_CONTOUR = os.path.join(DATA_PATH, 'sgpmet*20190508*.cdf')
EXAMPLE_CEIL_WILDCARD = os.path.join(DATA_PATH, 'sgpceil*.cdf')
EXAMPLE_TWP_SONDE_WILDCARD = os.path.join(DATA_PATH, 'twpsondewnpn*.cdf')
EXAMPLE_ANL_CSV = os.path.join(DATA_PATH, 'anltwr_mar19met.data')
EXAMPLE_VISST = os.path.join(
    DATA_PATH, 'twpvisstgridirtemp.c1.20050705.002500.nc')
EXAMPLE_MPL_1SAMPLE = os.path.join(DATA_PATH,
                                   'sgpmplpolfsC1.b1.20190502.000000.cdf')
EXAMPLE_IRT25m20s = os.path.join(DATA_PATH,
                                 'sgpirt25m20sC1.a0.20190601.000000.cdf')
