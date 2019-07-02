"""
act.qc.filter
=====================
Functions and methods for creating ancillary quality control variables
and filters (masks) which can be used with various corrections
routines in ACT.

"""

import numpy as np
import xarray as xr


@xr.register_dataset_accessor('qcfilter')
class QCFilter(object):
    """
    A class for building quality control variables containing arrays for
    filtering data based on a set of test condition typically based on the
    values in the data fields. These filters can be used in various
    algorithms and calculations within ACT.

    Attributes
    ----------
    check_for_ancillary_qc
        Method to check if a quality control variable exist in the dataset
        and return the quality control varible name.
        Will call create_qc_variable() to make variable if does not exist
        and update_ancillary_variable() to ensure linkage between data and
        quality control variable.
    create_qc_variable
        Method to create a new quality control variable in the dataset
        and set default values of 0.
    update_ancillary_variable
        Method to check if variable attribute ancillary_variables
        is set and if not to set to correct quality control variable name.
    add_test
        Method to add a test/filter to quality control variable
    remove_test
        Method to remove a test/filter from qualty control variable
    set_test
        Method to set a test/filter in a quality control variable
    unset_test
        Method to unset a test/filter in a quality control variable
    available_bit
        Method to get next availble bit or flag value for adding
        a new test to quality control variable.
    get_qc_test_mask
        Returns a numpy array False or True where a particular
        flag or bit is set in a numpy array. Options to return index array.

    """

    def __init__(self, xarray_obj):
        """ initialize """
        self._obj = xarray_obj

    def check_for_ancillary_qc(self, var_name):
        '''
        Method to check for corresponding quality control variable in
        the dataset. If it does not exist will create and correctly
        link the data variable to quality control variable with
        ancillary_variables attribute.

        Parameters
        ----------
        var_name : str
            data variable name

        Returns
        -------
        qc_var_name : str
            Name of existing or new quality control variable.

        '''
        qc_var_name = None
        try:
            ancillary_variables = \
                    self._obj[var_name].attrs['ancillary_variables']
            if isinstance(ancillary_variables, str):
                ancillary_variables = ancillary_variables.split()

            for var in ancillary_variables:
                for attr, value in self._obj[var].attrs.items():
                    if attr == 'standard_name' and 'quality_flag' in value:
                        qc_var_name = var

            if qc_var_name is None:
                qc_var_name = self._obj.qcfilter.create_qc_variable(var_name)

        except KeyError:
            # Since no ancillary_variables exist look for ARM style of QC
            # variable name. If it exists use it else create new
            # QC varaible.
            try:
                self._obj['qc_' + var_name]
                qc_var_name = 'qc_' + var_name
            except KeyError:
                qc_var_name = self._obj.qcfilter.create_qc_variable(var_name)

        # Make sure data varaible has a variable attribute linking
        # data variable to QC variable.
        self._obj.qcfilter.update_ancillary_variable(var_name, qc_var_name)

        return qc_var_name

    def create_qc_variable(self, var_name, flag_type=False,
                           flag_values_set_value=0):
        '''
        Method to create a quality control variable in the dataset.
        Will try not to destroy the qc variable by appending numbers
        to the variable name if needed.

        Parameters
        ----------
        var_name : str
            data variable name
        flag_type : boolean
            If an integer flag type should be created instead of
            bitpacked mask type. Will create flag_values instead of
            flag_masks.
        flag_values_set_value : int
            Initial flag value to use when initializing array.

        Returns
        -------
        qc_var_name : str
            Name of new quality control variable created.

        '''

        variable = self._obj[var_name].values

        # Make QC variable long name. The variable long_name attribute
        # may not exist so catch that error and set to default.
        try:
            qc_variable_long_name = ('Quality check results on field: ' +
                                     self._obj[var_name].attrs['long_name'])
        except KeyError:
            qc_variable_long_name = 'Quality check results for ' + var_name

        # Make a new quality control variable name. Check if exists in the
        # dataset. If so loop through creation of new name until one is
        # found that will not replace existing variable.
        qc_var_name = 'qc_' + var_name
        variable_names = list(self._obj.data_vars)
        if qc_var_name in variable_names:
            for ii in range(1, 100):
                temp_qc_var_name = '_'.join([qc_var_name, str(ii)])
                if temp_qc_var_name not in variable_names:
                    qc_var_name = temp_qc_var_name
                    break

        # Create the QC variable filled with 0 values matching the
        # shape of data variable. Add requried variable attributes.
        self._obj[qc_var_name] = (self._obj[var_name].dims,
                                  np.zeros(variable.shape, dtype=np.int32))

        # Update if using flag_values and don't want 0 to be default value.
        if flag_type and flag_values_set_value != 0:
            self._obj[qc_var_name].values = \
                self._obj[qc_var_name].values + int(flag_values_set_value)

        self._obj[qc_var_name].attrs['long_name'] = qc_variable_long_name
        self._obj[qc_var_name].attrs['units'] = '1'
        if flag_type:
            self._obj[qc_var_name].attrs['flag_values'] = []
        else:
            self._obj[qc_var_name].attrs['flag_masks'] = []
        self._obj[qc_var_name].attrs['flag_meanings'] = []
        self._obj[qc_var_name].attrs['flag_assessments'] = []
        self._obj[qc_var_name].attrs['standard_name'] = 'quality_flag'

        return qc_var_name

    def update_ancillary_variable(self, var_name, qc_var_name):
        '''
        Method to check if ancillary_variables varible attribute
        is set with quality control variable name.

        Parameters
        ----------
        var_name : str
            data variable name
        qc_var_name : str
            quality control variable name

        '''

        try:
            ancillary_variables = \
                self._obj[var_name].attrs['ancillary_variables']
            if qc_var_name not in ancillary_variables:

                ancillary_variables = ' '.join([ancillary_variables,
                                                qc_var_name])
        except KeyError:
            ancillary_variables = qc_var_name

        self._obj[var_name].attrs['ancillary_variables'] = ancillary_variables

    def add_test(self, var_name, index=None, test_number=None,
                 test_meaning=None, test_assessment='Bad',
                 flag_value=False):
        '''
        Method to add a new test/filter to a qualit control variable.

        Parameters
        ----------
        var_name : str
            data variable name
        index : int or list of int or numpy array
            Indexes into quality control array to set the test bit.
            If not set or set to None will not set the test on any
            element of the quality control variable but will still
            add the test to the flag_masks, flag_meanings and
            flag_assessments attributes.
        test_number : int
            Test number to use. If keyword is not set will use first
            available test bit/test number.
        test_meaning : str
            String describing the test. Will be added to flag_meanings
            variable attribute.
        test_assessment : str
            String describing the test assessment. If not set will use
            "Bad" as the string to append to flag_assessments. Will
            update to be lower case and then capitalized.
        flag_value : boolean
            Switch to use flag_values integer quality control.

        Returns
        -------
        test_dict : dictionary
            A dictionary containing information added to the QC
            variable.

        '''

        test_dict = {}

        if test_meaning is None:
            raise ValueError('You need to provide a value for test_meaning '
                             'keyword when calling the add_test method')

        # This ensures the indexing will work even if given float values.
        if index is not None:
            index = np.array(index, dtype=np.int)

        # Ensure assessment is lowercase and capitalized to be consistent
        test_assessment = test_assessment.lower().capitalize()

        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if test_number is None:
            test_number = self._obj.qcfilter.available_bit(
                qc_var_name)

        self._obj.qcfilter.set_test(var_name, index, test_number,
                                    flag_value)

        if flag_value:
            flag_values = self._obj[qc_var_name].attrs['flag_values']
            flag_values.append(test_number)
            self._obj[qc_var_name].attrs['flag_values'] = flag_values
        else:
            flag_masks = self._obj[qc_var_name].attrs['flag_masks']
            flag_masks.append(set_bit(0, test_number))
            self._obj[qc_var_name].attrs['flag_masks'] = flag_masks

        flag_meanings = self._obj[qc_var_name].attrs['flag_meanings']
        flag_meanings.append(test_meaning)
        self._obj[qc_var_name].attrs['flag_meanings'] = flag_meanings

        flag_assessments = self._obj[qc_var_name].attrs['flag_assessments']
        flag_assessments.append(test_assessment)
        self._obj[qc_var_name].attrs['flag_assessments'] = flag_assessments

        test_dict['test_number'] = test_number
        test_dict['test_meaning'] = test_meaning
        test_dict['test_assessment'] = test_assessment
        test_dict['qc_variable_name'] = qc_var_name

        return test_dict

    def remove_test(self, var_name, test_number=None, flag_value=False,
                    flag_values_reset_value=0):
        '''
        Method to remove a test/filter from a quality control variable.

        Parameters
        ----------
        var_name : str
            data variable name
        test_number : int
            Test number to remove
        flag_value : boolean
            Switch to use flag_values integer quality control
        flag_values_reset_value : int
            Value to use when resetting a flag_values value to not be set.

        '''

        if test_number is None:
            raise ValueError('You need to provide a value for test_number '
                             'keyword when calling the add_test method')

        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        # Determine which index is using the test number
        index = None
        if flag_value:
            flag_values = self._obj[qc_var_name].attrs['flag_values']
            for ii, flag_num in enumerate(flag_values):
                if flag_num == test_number:
                    index = ii
                    break
        else:
            flag_masks = self._obj[qc_var_name].attrs['flag_masks']
            for ii, bit_num in enumerate(flag_masks):
                if parse_bit(bit_num)[0] == test_number:
                    index = ii
                    break

        # If can't find the index of test return before doing anything.
        if index is None:
            return

        if flag_value:
            remove_index = self._obj.qcfilter.get_qc_test_mask(
                    var_name, test_number, return_index=True, flag_value=True)
            self._obj.qcfilter.unset_test(var_name, remove_index, test_number,
                                          flag_value, flag_values_reset_value)
            del flag_values[index]
            self._obj[qc_var_name].attrs['flag_values'] = flag_values
        else:
            remove_index = self._obj.qcfilter.get_qc_test_mask(
                    var_name, test_number, return_index=True)
            self._obj.qcfilter.unset_test(var_name, remove_index, test_number,
                                          flag_value, flag_values_reset_value)
            del flag_masks[index]
            self._obj[qc_var_name].attrs['flag_masks'] = flag_masks

        flag_meanings = self._obj[qc_var_name].attrs['flag_meanings']
        del flag_meanings[index]
        self._obj[qc_var_name].attrs['flag_meanings'] = flag_meanings

        flag_assessments = self._obj[qc_var_name].attrs['flag_assessments']
        del flag_assessments[index]
        self._obj[qc_var_name].attrs['flag_assessments'] = flag_assessments

    def set_test(self, var_name, index=None, test_number=None,
                 flag_value=False):
        '''
        Method to set a test/filter in a quality control variable.

        Parameters
        ----------
        var_name : str
            data variable name
        index : int or list or numpy array
            Index to set test in quality control array. If want to
            unset all values will need to pass in index of all values.
        test_number : int
            Test number to remove
        flag_value : boolean
            Switch to use flag_values integer quality control

        '''

        if index is None:
            return

        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        qc_variable = self._obj[qc_var_name].values
        if index is not None:
            if flag_value:
                qc_variable[index] = test_number
            else:
                qc_variable[index] = set_bit(qc_variable[index], test_number)

        self._obj[qc_var_name].values = qc_variable

    def unset_test(self, var_name, index=None, test_number=None,
                   flag_value=False, flag_values_reset_value=0):
        '''
        Method to unset a test/filter from a quality control variable.

        Parameters
        ----------
        var_name : str
            data variable name
        index : int or list or numpy array
            Index to unset test in quality control array. If want to
            unset all values will need to pass in index of all values.
        test_number : int
            Test number to remove
        flag_value : boolean
            Switch to use flag_values integer quality control
        flag_values_reset_value : int
            Value to use when resetting a flag_values value to not be set.

        '''

        if index is None:
            return

        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        qc_variable = self._obj[qc_var_name].values
        if flag_value:
            qc_variable[index] = flag_values_reset_value
        else:
            qc_variable[index] = unset_bit(qc_variable[index], test_number)

        self._obj[qc_var_name].values = qc_variable

    def available_bit(self, qc_var_name):
        '''
        Method to determine next availble bit or flag to use with a QC test.
        Will check for flag_masks first and if not found will check for
        flag_values. This will drive how the next value is chosen.

        Parameters
        ----------
        qc_var_name : str
            quality control variable name

        Returns
        -------
        test_num : int
            Next available test number

        '''

        try:
            flag_masks = self._obj[qc_var_name].attrs['flag_masks']
            flag_value = False
        except KeyError:
            flag_masks = self._obj[qc_var_name].attrs['flag_values']
            flag_value = True

        if flag_masks == []:
            next_bit = 1
        else:
            if flag_value:
                next_bit = max(flag_masks) + 1
            else:
                next_bit = parse_bit(max(flag_masks))[0] + 1

        return next_bit

    def get_qc_test_mask(self, var_name, test_number, flag_value=False,
                         return_index=False):
        """
        Returns a numpy array of False or True where a particular
        flag or bit is set in a numpy array.

        Parameters
        ----------
        var_name : str
            data variable name
        test_number : int
            Test number to return array where test is set
        flag_value : boolean
            Switch to use flag_values integer quality control
        return_index : boolean
            Return a numpy array of index numbers into QC array where the
            test is set instead of 0 or 1 mask.

        Returns : numpy int array
        --------
            An array of 0 or 1 (False or True) where the test number or
            bit was set.

         Example
        --------
            > from act.io.armfiles import read_netcdf
            > from act.tests import EXAMPLE_IRT25m20s
            > ds_object = read_netcdf(EXAMPLE_IRT25m20s)
            > var_name = 'inst_up_long_dome_resist'
            > result = ds_object.qcfilter.add_test(var_name,
                index=[0,1,2],test_meaning='Birds!')
            > qc_var_name = result['qc_variable_name']
            > mask = ds_object.qcfilter.get_qc_test_mask(var_name,
                result['test_number'], return_index=True)
            > mask
            array([0, 1, 2])

            > mask = ds_object.qcfilter.get_qc_test_mask(var_name,
                result['test_number'])
            > mask
            array([ True,  True,  True, ..., False, False, False])

            > data = ds_object[var_name].values
            > data[mask]
            array([7.84  , 7.8777, 7.8965], dtype=float32)

            > import numpy as np
            > data[mask] = np.nan
            > data
            array([   nan,    nan,    nan, ..., 7.6705, 7.6892, 7.6892],
                dtype=float32)


        """

        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        qc_variable = self._obj[qc_var_name].values

        if flag_value:
            tripped = np.where(qc_variable == test_number)
        else:
            check_bit = set_bit(0, test_number) & qc_variable
            tripped = np.where(check_bit > 0)

        test_mask = np.zeros(qc_variable.shape, dtype='int')
        # Make sure test_mask is an array. If qc_variable is scalar will
        # be retuned from np.zeros as scalar.
        test_mask = np.atleast_1d(test_mask)
        test_mask[tripped] = 1
        test_mask = np.ma.make_mask(test_mask)

        if return_index:
            test_mask = np.where(test_mask)[0]

        return test_mask


def set_bit(array, bit_number):
    '''
    Function to set a quality control bit given an a scalar or
    array of values and a bit number.

    Parameters
    ----------
    array: int or numpy array
        The bitpacked array to set the bit number.
    bit_number: int
        The bit (or test) number to set

    Returns: int, numpy array, tuple, list
    --------
        Integer or numpy array with bit set for each element of the array.
        Returned in same type.

    Example:
    --------
        Example use setting bit 2 to an array called data:

        > data = np.array(range(0,7))
        > data = set_bit(data,2)
        > data
        array([2, 3, 2, 3, 6, 7, 6])

    '''
    was_list = False
    was_tuple = False
    if isinstance(array, list):
        array = np.array(array)
        was_list = True

    if isinstance(array, tuple):
        array = np.array(array)
        was_tuple = True

    if bit_number > 0:
        array |= (1 << bit_number - 1)

    if was_list:
        array = list(array)

    if was_tuple:
        array = tuple(array)

    return array


def unset_bit(array, bit_number):
    '''
    Function to remove a quality control bit given a
    scalar or array of values and a bit number.

    Parameters
    ----------
    array: int or numpy array
        Array of integers containing bit packed numbers.
    bit_number: int
        Bit number to remove.

    Returns: int or numpy array
    --------
        Returns same data type as array entered with bit removed. Will
        fail gracefully if the bit requested to be removed was not set.

    Example:
    --------
       Example use removing bit 2 from an array called data:

       > data = set_bit(0,2)
       > data = set_bit(data,3)
       > data
       6

       > data = unset_bit(data,2)
       > data
       4

    '''

    was_list = False
    was_tuple = False
    if isinstance(array, list):
        array = np.array(array)
        was_list = True

    if isinstance(array, tuple):
        array = np.array(array)
        was_tuple = True

    if bit_number > 0:
        array = array & ~ (1 << bit_number - 1)

    if was_list:
        array = list(array)

    if was_tuple:
        array = tuple(array)

    return array


def parse_bit(qc_bit):
    '''
    Given a single integer value, return bit positions.

    Parameters
    ----------
    qc_bit: int or numpy int
        Bit packed integer number to be parsed.

    Returns: numpy.int32 array
    --------
        Array containing all bit numbers of the bit packed number.
        If no bits set returns empty array.

    Example:
    --------
        > parse_bit(7)
        array([1, 2, 3])

    '''

    if isinstance(qc_bit, (list, tuple, np.ndarray)):
        if len(qc_bit) > 1:
            raise ValueError("Must be a single value.")
        qc_bit = qc_bit[0]

    if qc_bit < 0:
        raise ValueError("Must be a positive integer.")

    bit_number = []
#    if qc_bit == 0:
#        bit_number.append(0)

    counter = 0
    while qc_bit > 0:
        temp_value = qc_bit % 2
        qc_bit = qc_bit >> 1
        counter += 1
        if temp_value == 1:
            bit_number.append(counter)

    bit_number = np.asarray(bit_number, dtype=np.int32)

    return bit_number
