"""
===========================================================
Example for working with embedded quality control variables
===========================================================

This is an example of how to use existing or create new quality
control varibles.

"""

from act.io.armfiles import read_netcdf
from act.tests import EXAMPLE_IRT25m20s
import numpy as np

# Read a data file that does not have any embedded quality control
# variables. This data comes from the example dataset within ACT.
ds_object = read_netcdf(EXAMPLE_IRT25m20s)

# The name of the data variable we wish to work with
var_name = 'inst_up_long_dome_resist'

# Since there is no embedded quality control varible one will be
# created for us.
# Perform adding of quality control variables to object.
# We can start with adding where the data are set to missing value.
# First we will change the first value to NaN to simulate where
# a missing value exist in the data file.
data = ds_object[var_name].values
data[0] = np.nan
ds_object[var_name].values = data

# Add a test for where the data are set to missing value.
result = ds_object.qcfilter.add_missing_value_test(var_name)

# The returned dictionary will contain the information added to the
# quality control varible for direct use now. Or the information
# can be looked up later for use.
print('result= ', result)
print()

# We can add a second test where data is less than a specified value.
result = ds_object.qcfilter.add_less_test(var_name, 7.8)

# Next we add a test to indicate where a value is greater than
# or equal to a specified number. We also set the assessement
# to a user defined word. The default assessment is "Bad".
result = ds_object.qcfilter.add_greater_equal_test(var_name, 12,
                                                   test_assessment='Suspect')

# We can now get the data as a numpy masked array with a mask
# where the third test we added (greater than or equal to) using
# the result dictionary to get the test number created for us.
data = ds_object.qcfilter.get_masked_data(var_name,
                                          rm_tests=result['test_number'])
print('Data type=', type(data))
print()

# Or we can get the masked array for all tests that use the assessment
# set to "Bad".
data = ds_object.qcfilter.get_masked_data(var_name, rm_assessments=['Bad'])

# If we prefer to mask all data for both Bad or Suspect we can list
# as many assessments as needed
data = ds_object.qcfilter.get_masked_data(var_name,
                                          rm_assessments=['Suspect', 'Bad'])
print('data=', data)
print()

# We can create our own test by creating an array of indexes of where
# we want the test to be set and call the method to create our own test.
data = ds_object.qcfilter.get_masked_data(var_name)
diff = np.diff(data)
max_difference = 0.04
index = np.where(diff > max_difference)[0]
result = ds_object.qcfilter.add_test(
    var_name, index=index,
    test_meaning='Difference is greater than {}'.format(str(max_difference)),
    test_assessment='Suspect')

# If we prefer to work with numpy arrays directly we can get the
# data array converted to a numpy array with masked values set
# to NaN.
data = ds_object.qcfilter.get_masked_data(var_name,
                                          rm_assessments=['Suspect', 'Bad'],
                                          return_nan_array=True)
print('Data type=', type(data))
print('data=', data)
print()

# We can see how the quality control data is stored and what assessments,
# or test descriptions are set. The tests have also added attributes to
# store the test limit values.
print('QC Variable=', ds_object[result['qc_variable_name']])
