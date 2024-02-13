"""
Sample data file for use in testing. These files should only
be used for testing ACT.

"""


from arm_test_data import DATASETS

# Single files
EXAMPLE_MET1 = DATASETS.fetch('sgpmetE13.b1.20190101.000000.cdf')
EXAMPLE_MET_SAIL = DATASETS.fetch('gucmetM1.b1.20230301.000000.cdf')
EXAMPLE_MET_CSV = DATASETS.fetch('sgpmetE13.b1.20210401.000000.csv')
EXAMPLE_METE40 = DATASETS.fetch('sgpmetE40.b1.20190508.000000.cdf')
EXAMPLE_CEIL1 = DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc')
EXAMPLE_SONDE1 = DATASETS.fetch('sgpsondewnpnC1.b1.20190101.053200.cdf')
EXAMPLE_LCL1 = DATASETS.fetch('met_lcl.nc')
EXAMPLE_ANL_CSV = DATASETS.fetch('anltwr_mar19met.data')
EXAMPLE_VISST = DATASETS.fetch('twpvisstgridirtemp.c1.20050705.002500.nc')
EXAMPLE_MPL_1SAMPLE = DATASETS.fetch('sgpmplpolfsC1.b1.20190502.000000.cdf')
EXAMPLE_IRT25m20s = DATASETS.fetch('sgpirt25m20sC1.a0.20190601.000000.cdf')
EXAMPLE_NAV = DATASETS.fetch('marnavM1.a1.20180201.000000.nc')
EXAMPLE_AOSMET = DATASETS.fetch('maraosmetM1.a1.20180201.000000.nc')
EXAMPLE_DLPPI = DATASETS.fetch('sgpdlppiC1.b1.20191015.120023.cdf')
EXAMPLE_BRS = DATASETS.fetch('sgpbrsC1.b1.20190705.000000.cdf')
EXAMPLE_AERI = DATASETS.fetch('sgpaerich1C1.b1.20190501.000342.nc')
EXAMPLE_IRTSST = DATASETS.fetch('marirtsstM1.b1.20190320.000000.nc')
EXAMPLE_MFRSR = DATASETS.fetch('sgpmfrsr7nchE11.b1.20210329.070000.nc')
EXAMPLE_SURFSPECALB1MLAWER = DATASETS.fetch('nsasurfspecalb1mlawerC1.c1.20160609.080000.nc')
EXAMPLE_SIGMA_MPLV5 = DATASETS.fetch('201509021500.bi')
EXAMPLE_RL1 = DATASETS.fetch('sgprlC1.a0.20160131.000000.nc')
EXAMPLE_CO2FLX4M = DATASETS.fetch('sgpco2flx4mC1.b1.20201007.001500.nc')
EXAMPLE_SIRS = DATASETS.fetch('sgpsirsE13.b1.20190101.000000.cdf')
EXAMPLE_GML_RADIATION = DATASETS.fetch('brw21001.dat')
EXAMPLE_GML_MET = DATASETS.fetch('met_brw_insitu_1_obop_hour_2020.txt')
EXAMPLE_GML_OZONE = DATASETS.fetch('brw_12_2020_hour.dat')
EXAMPLE_GML_CO2 = DATASETS.fetch('co2_brw_surface-insitu_1_ccgg_MonthlyData.txt')
EXAMPLE_GML_HALO = DATASETS.fetch('brw_CCl4_Day.dat')
EXAMPLE_MET_TEST1 = DATASETS.fetch('sgpmet_no_time.nc')
EXAMPLE_MET_TEST2 = DATASETS.fetch('sgpmet_test_time.nc')
EXAMPLE_NOAA_PSL = DATASETS.fetch('ctd21125.15w')
EXAMPLE_NOAA_PSL_TEMPERATURE = DATASETS.fetch('ctd22187.00t.txt')
EXAMPLE_SP2B = DATASETS.fetch('mosaossp2M1.00.20191216.130601.raw.20191216x193.sp2b')
EXAMPLE_INI = DATASETS.fetch('mosaossp2M1.00.20191216.000601.raw.20191216000000.ini')
EXAMPLE_HK = DATASETS.fetch('mosaossp2auxM1.00.20191217.010801.raw.20191216000000.hk')
EXAMPLE_MET_YAML = DATASETS.fetch('sgpmetE13.b1.yaml')
EXAMPLE_CLOUDPHASE = DATASETS.fetch('nsacloudphaseC1.c1.20180601.000000.nc')
EXAMPLE_AAF_ICARTT = DATASETS.fetch('AAFNAV_COR_20181104_R0.ict')
EXAMPLE_NEON = DATASETS.fetch(
    'NEON.D18.BARR.DP1.00002.001.000.010.001.SAAT_1min.2022-10.expanded.20221107T205629Z.csv'
)
EXAMPLE_NEON_VARIABLE = DATASETS.fetch('NEON.D18.BARR.DP1.00002.001.variables.20221201T110553Z.csv')
EXAMPLE_NEON_POSITION = DATASETS.fetch(
    'NEON.D18.BARR.DP1.00002.001.sensor_positions.20221107T205629Z.csv'
)
EXAMPLE_DOD = DATASETS.fetch('vdis.b1')
EXAMPLE_EBBR1 = DATASETS.fetch('sgp30ebbrE32.b1.20191125.000000.nc')
EXAMPLE_EBBR2 = DATASETS.fetch('sgp30ebbrE32.b1.20191130.000000.nc')
EXAMPLE_EBBR3 = DATASETS.fetch('sgp30ebbrE13.b1.20190601.000000.nc')
EXAMPLE_ECOR = DATASETS.fetch('sgp30ecorE14.b1.20190601.000000.cdf')
EXAMPLE_SEBS = DATASETS.fetch('sgpsebsE14.b1.20190601.000000.cdf')
EXAMPLE_MFAS_SODAR = DATASETS.fetch('sodar.20230404.mnd')
EXAMPLE_ENA_MET = DATASETS.fetch('enametC1.b1.20221109.000000.cdf')
EXAMPLE_CCN = DATASETS.fetch('sgpaosccn2colaE13.b1.20170903.000000.nc')
EXAMPLE_OLD_QC = DATASETS.fetch('sgp30ecorE6.b1.20040705.000000.cdf')
EXAMPLE_SONDE_WILDCARD = DATASETS.fetch('sgpsondewnpnC1.b1.20190101.053200.cdf')
EXAMPLE_CEIL_WILDCARD = DATASETS.fetch('sgpceilC1.b1.20190101.000000.nc')
EXAMPLE_HYSPLIT = DATASETS.fetch('houstonaug300.0summer2010080100')

# Multiple files in a list
dlppi_multi_list = ['sgpdlppiC1.b1.20191015.120023.cdf', 'sgpdlppiC1.b1.20191015.121506.cdf']
EXAMPLE_DLPPI_MULTI = [DATASETS.fetch(file) for file in dlppi_multi_list]
noaa_psl_list = ['ayp22199.21m', 'ayp22200.00m']
EXAMPLE_NOAA_PSL_SURFACEMET = [DATASETS.fetch(file) for file in noaa_psl_list]
met_wildcard_list = [
    'sgpmetE13.b1.20190101.000000.cdf',
    'sgpmetE13.b1.20190102.000000.cdf',
    'sgpmetE13.b1.20190103.000000.cdf',
    'sgpmetE13.b1.20190104.000000.cdf',
    'sgpmetE13.b1.20190105.000000.cdf',
    'sgpmetE13.b1.20190106.000000.cdf',
    'sgpmetE13.b1.20190107.000000.cdf',
]
EXAMPLE_MET_WILDCARD = [DATASETS.fetch(file) for file in met_wildcard_list]
met_contour_list = [
    'sgpmetE15.b1.20190508.000000.cdf',
    'sgpmetE31.b1.20190508.000000.cdf',
    'sgpmetE32.b1.20190508.000000.cdf',
    'sgpmetE33.b1.20190508.000000.cdf',
    'sgpmetE34.b1.20190508.000000.cdf',
    'sgpmetE35.b1.20190508.000000.cdf',
    'sgpmetE36.b1.20190508.000000.cdf',
    'sgpmetE37.b1.20190508.000000.cdf',
    'sgpmetE38.b1.20190508.000000.cdf',
    'sgpmetE39.b1.20190508.000000.cdf',
    'sgpmetE40.b1.20190508.000000.cdf',
    'sgpmetE9.b1.20190508.000000.cdf',
    'sgpmetE13.b1.20190508.000000.cdf',
]
EXAMPLE_MET_CONTOUR = [DATASETS.fetch(file) for file in met_contour_list]
twp_sonde_wildcard_list = [
    'twpsondewnpnC3.b1.20060119.050300.custom.cdf',
    'twpsondewnpnC3.b1.20060119.112000.custom.cdf',
    'twpsondewnpnC3.b1.20060119.163300.custom.cdf',
    'twpsondewnpnC3.b1.20060119.231600.custom.cdf',
    'twpsondewnpnC3.b1.20060120.043800.custom.cdf',
    'twpsondewnpnC3.b1.20060120.111900.custom.cdf',
    'twpsondewnpnC3.b1.20060120.170800.custom.cdf',
    'twpsondewnpnC3.b1.20060120.231500.custom.cdf',
    'twpsondewnpnC3.b1.20060121.051500.custom.cdf',
    'twpsondewnpnC3.b1.20060121.111600.custom.cdf',
    'twpsondewnpnC3.b1.20060121.171600.custom.cdf',
    'twpsondewnpnC3.b1.20060121.231600.custom.cdf',
    'twpsondewnpnC3.b1.20060122.052600.custom.cdf',
    'twpsondewnpnC3.b1.20060122.111500.custom.cdf',
    'twpsondewnpnC3.b1.20060122.171800.custom.cdf',
    'twpsondewnpnC3.b1.20060122.232600.custom.cdf',
    'twpsondewnpnC3.b1.20060123.052500.custom.cdf',
    'twpsondewnpnC3.b1.20060123.111700.custom.cdf',
    'twpsondewnpnC3.b1.20060123.171600.custom.cdf',
    'twpsondewnpnC3.b1.20060123.231500.custom.cdf',
    'twpsondewnpnC3.b1.20060124.051500.custom.cdf',
    'twpsondewnpnC3.b1.20060124.111800.custom.cdf',
    'twpsondewnpnC3.b1.20060124.171700.custom.cdf',
    'twpsondewnpnC3.b1.20060124.231500.custom.cdf',
]
EXAMPLE_TWP_SONDE_WILDCARD = [DATASETS.fetch(file) for file in twp_sonde_wildcard_list]
twp_sonde_20060121_list = [
    'twpsondewnpnC3.b1.20060121.051500.custom.cdf',
    'twpsondewnpnC3.b1.20060121.111600.custom.cdf',
    'twpsondewnpnC3.b1.20060121.171600.custom.cdf',
    'twpsondewnpnC3.b1.20060121.231600.custom.cdf',
]
EXAMPLE_TWP_SONDE_20060121 = [DATASETS.fetch(file) for file in twp_sonde_20060121_list]
stamp_wildcard_list = [
    'sgpstampE13.b1.20200101.000000.nc',
    'sgpstampE31.b1.20200101.000000.nc',
    'sgpstampE32.b1.20200101.000000.nc',
    'sgpstampE33.b1.20200101.000000.nc',
    'sgpstampE34.b1.20200101.000000.nc',
    'sgpstampE9.b1.20200101.000000.nc',
]
EXAMPLE_STAMP_WILDCARD = [DATASETS.fetch(file) for file in stamp_wildcard_list]
mmcr_list = ['sgpmmcrC1.b1.1.cdf', 'sgpmmcrC1.b1.2.cdf']
EXAMPLE_MMCR = [DATASETS.fetch(file) for file in mmcr_list]
