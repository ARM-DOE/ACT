from act.io.armfiles import read_netcdf
from act.tests import EXAMPLE_MET1

ds = read_netcdf(EXAMPLE_MET1)
ds.clean.cleanup()

var_name = 'atmos_pressure'

ds_1 = ds.mean()

ds.qcfilter.add_less_test(var_name, 99, test_assessment='Bad')
ds.qcfilter.datafilter(rm_assessments='Bad')
ds_2 = ds.mean()
print('All data: ', ds_1[var_name].values)
print('Bad Removed: ', ds_2[var_name].values)
