"""
Working with embedded quality control variables
-----------------------------------------------

This is an example of how to use existing or create new quality
control varibles. All the tests are located in act/qc/qctests.py
file but called under the qcfilter method.

"""

from arm_test_data import DATASETS
import numpy as np

from act.io.arm import read_arm_netcdf
from act.qc.qcfilter import parse_bit

# Read a data file that does not have any embedded quality control
# variables. This data comes from the example dataset within ACT.
# Can also read data that has existing quality control variables
# and add, manipulate or use those variables the same.
filename_irt = DATASETS.fetch('sgpirt25m20sC1.a0.20190601.000000.cdf')
ds = read_arm_netcdf(filename_irt)

# The name of the data variable we wish to work with
var_name = 'inst_up_long_dome_resist'

# Since there is no embedded quality control varible one will be
# created for us.
# We can start with adding where the data are set to missing value.
# First we will change the first value to NaN to simulate where
# a missing value exist in the data file.
data = ds[var_name].values
data[0] = np.nan
ds[var_name].values = data

# Add a test for where the data are set to missing value.
# Since a quality control variable does not exist in the file
# one will be created as part of adding this test.
result = ds.qcfilter.add_missing_value_test(var_name)

# The returned dictionary will contain the information added to the
# quality control varible for direct use now. Or the information
# can be looked up later for use.
print('\nresult =', result)

# We can add a second test where data is less than a specified value.
result = ds.qcfilter.add_less_test(var_name, 7.8)

# Next we add a test to indicate where a value is greater than
# or equal to a specified number. We also set the assessement
# to a user defined word. The default assessment is "Bad".
result = ds.qcfilter.add_greater_equal_test(var_name, 12, test_assessment='Suspect')

# We can now get the data as a numpy masked array with a mask set
# where the third test we added (greater than or equal to) using
# the result dictionary to get the test number created for us.
data = ds.qcfilter.get_masked_data(var_name, rm_tests=result['test_number'])
print('\nData type =', type(data))

# Or we can get the masked array for all tests that use the assessment
# set to "Bad".
data = ds.qcfilter.get_masked_data(var_name, rm_assessments=['Bad'])

# If we prefer to mask all data for both Bad or Suspect we can list
# as many assessments as needed
data = ds.qcfilter.get_masked_data(var_name, rm_assessments=['Suspect', 'Bad'])
print('\ndata =', data)

# We can convert the masked array into numpy array and choose the fill value.
data = data.filled(fill_value=np.nan)
print('\ndata filled with masked array fill_value =', data)

# We can create our own test by creating an array of indexes of where
# we want the test to be set and call the method to create our own test.
# We can allow the method to pick the test number (next available)
# or set the test number we wan to use. This example uses test number
# 5 to demonstrate how not all tests need to be used in order.
data = ds.qcfilter.get_masked_data(var_name)
diff = np.diff(data)
max_difference = 0.04
data = np.ma.masked_greater(diff, max_difference)
index = np.where(data.mask)[0]
result = ds.qcfilter.add_test(
    var_name,
    index=index,
    test_meaning=f'Difference is greater than {max_difference}',
    test_assessment='Suspect',
    test_number=5,
)

# If we prefer to work with numpy arrays directly we can return the
# data array converted to a numpy array with masked values set
# to NaN. Here we are requesting both Suspect and Bad data be masked.
data = ds.qcfilter.get_masked_data(
    var_name, rm_assessments=['Suspect', 'Bad'], return_nan_array=True
)
print('\nData type =', type(data))
print('data =', data)

# We can see how the quality control data is stored and what assessments,
# or test descriptions are set. Some of the tests have also added attributes to
# store the test limit values.
qc_variable = ds[result['qc_variable_name']]
print('\nQC Variable =', qc_variable)

# The test numbers are not the flag_masks numbers. The flag masks numbers
# are bit-paked numbers used to store what bit is set. To see the test
# numbers we can unpack the bits.
print('\nmask : test')
print('-' * 11)
for mask in qc_variable.attrs['flag_masks']:
    print(mask, ' : ', parse_bit(mask))

# We can also just use the get_masked_data() method to get data
# the same as using ".values" method on the xarray dataset. If we don't
# request any tests or assessments to mask the returned masked array
# will not have any mask set. The returned value is a numpy masked array
# where the raw numpy array is accessable with .data property.
data = ds.qcfilter.get_masked_data(var_name)
print('\nNormal numpy array data values:', data.data)
print('Mask associated with values:', data.mask)

# We can use the get_masked_data() method to return a masked array
# where the test is set in the quality control varialbe, and use the
# masked array method to see if any of the values have the test set.
data = ds.qcfilter.get_masked_data(var_name, rm_tests=3)
print('\nAt least one less than test set =', data.mask.any())
data = ds.qcfilter.get_masked_data(var_name, rm_tests=4)
print('At least one difference test set =', data.mask.any())
