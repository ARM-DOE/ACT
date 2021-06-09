"""
Sample data file for use in testing. These files should only
be used for testing ACT.

"""

import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

EXAMPLE_MET1 = os.path.join(DATA_PATH, 'sgpmetE13.b1.20190101.000000.cdf')
EXAMPLE_MET_CSV = os.path.join(DATA_PATH, 'sgpmetE13.*csv')
EXAMPLE_METE40 = os.path.join(DATA_PATH, 'sgpmetE40.b1.20190508.000000.cdf')
EXAMPLE_CEIL1 = os.path.join(DATA_PATH, 'sgpceilC1.b1.20190101.000000.nc')
EXAMPLE_SONDE1 = os.path.join(DATA_PATH,
                              'sgpsondewnpnC1.b1.20190101.053200.cdf')
EXAMPLE_LCL1 = os.path.join(DATA_PATH, 'met_lcl.nc')
EXAMPLE_SONDE_WILDCARD = os.path.join(DATA_PATH, 'sgpsondewnpn*.cdf')
EXAMPLE_MET_WILDCARD = os.path.join(DATA_PATH, 'sgpmet*201901*.cdf')
EXAMPLE_MET_CONTOUR = os.path.join(DATA_PATH, 'sgpmet*20190508*.cdf')
EXAMPLE_CEIL_WILDCARD = os.path.join(DATA_PATH, 'sgpceil*.cdf')
EXAMPLE_TWP_SONDE_WILDCARD = os.path.join(DATA_PATH, 'twpsondewnpn*.cdf')
EXAMPLE_TWP_SONDE_20060121 = os.path.join(DATA_PATH, 'twpsondewnpn*20060121*.cdf')
EXAMPLE_ANL_CSV = os.path.join(DATA_PATH, 'anltwr_mar19met.data')
EXAMPLE_VISST = os.path.join(
    DATA_PATH, 'twpvisstgridirtemp.c1.20050705.002500.nc')
EXAMPLE_MPL_1SAMPLE = os.path.join(DATA_PATH,
                                   'sgpmplpolfsC1.b1.20190502.000000.cdf')
EXAMPLE_IRT25m20s = os.path.join(DATA_PATH,
                                 'sgpirt25m20sC1.a0.20190601.000000.cdf')
EXAMPLE_NAV = os.path.join(DATA_PATH,
                           'marnavM1.a1.20180201.000000.nc')
EXAMPLE_AOSMET = os.path.join(DATA_PATH,
                              'maraosmetM1.a1.20180201.000000.nc')
EXAMPLE_DLPPI = os.path.join(DATA_PATH, 'sgpdlppiC1.b1.20191015.120023.cdf')
EXAMPLE_DLPPI_MULTI = os.path.join(DATA_PATH, 'sgpdlppiC1.b1.20191015.*.cdf')
EXAMPLE_EBBR1 = os.path.join(DATA_PATH, 'sgp30ebbrE32.b1.20191125.000000.nc')
EXAMPLE_EBBR2 = os.path.join(DATA_PATH, 'sgp30ebbrE32.b1.20191130.000000.nc')
EXAMPLE_BRS = os.path.join(DATA_PATH, 'sgpbrsC1.b1.20190705.000000.cdf')
EXAMPLE_AERI = os.path.join(DATA_PATH, 'sgpaerich1C1.b1.20190501.000342.nc')
EXAMPLE_IRTSST = os.path.join(DATA_PATH, 'marirtsstM1.b1.20190320.000000.nc')
EXAMPLE_MFRSR = os.path.join(DATA_PATH, 'sgpmfrsr7nchE38.b1.20190514.180000.nc')
EXAMPLE_SURFSPECALB1MLAWER = os.path.join(
    DATA_PATH, 'nsasurfspecalb1mlawerC1.c1.20160609.080000.nc')
EXAMPLE_SIGMA_MPLV5 = os.path.join(DATA_PATH, '201509021500.bi')
EXAMPLE_RL1 = os.path.join(DATA_PATH, 'sgprlC1.a0.20160131.000000.nc')
EXAMPLE_CO2FLX4M = os.path.join(DATA_PATH, 'sgpco2flx4mC1.b1.20201007.001500.nc')
EXAMPLE_SIRS = os.path.join(DATA_PATH, 'sgpsirsE13.b1.20190101.000000.cdf')
EXAMPLE_GML_RADIATION = os.path.join(DATA_PATH, 'brw21001.dat')
EXAMPLE_GML_MET = os.path.join(DATA_PATH, 'met_brw_insitu_1_obop_hour_2020.txt')
EXAMPLE_GML_OZONE = os.path.join(DATA_PATH, 'brw_12_2020_hour.dat')
EXAMPLE_GML_CO2 = os.path.join(DATA_PATH, 'co2_brw_surface-insitu_1_ccgg_MonthlyData.txt')
EXAMPLE_GML_HALO = os.path.join(DATA_PATH, 'brw_CCl4_Day.dat')
EXAMPLE_MET_TEST1 = os.path.join(DATA_PATH, 'sgpmet_no_time.nc')
EXAMPLE_MET_TEST2 = os.path.join(DATA_PATH, 'sgpmet_test_time.nc')
EXAMPLE_STAMP_WILDCARD = os.path.join(DATA_PATH, 'sgpstamp*202001*.nc')
