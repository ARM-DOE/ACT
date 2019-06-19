"""
act.utils.qc_data_utils
------------------------

This module contains classes and functions written to handle embedded
quality control variables and ARM Data Quality Reports.

"""

import numpy as np
# import xarray as xr
# import collections
# import warnings

# from act.utils.data_utils import get_missing_value


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


def remove_bit(array, bit_number):
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

       > data = remove_bit(data,2)
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


def get_qc_test_mask(test_number, qc_data, bit=True, return_bool=False):
    """
    Returns a numpy array of 0 or 1 (False or True) where a particular
    flag or bit is set in a numpy array.

    Parameters
    ----------
    test_number : int
        Test number to compare with quality control data array to check.
    qc_data : numpy array
        Quality control array of flag or bitpacked valus to compare.
    bit : boolean
        Indicate if test_number is a bit test number (flag_masks method)
        not a flag number (flag_values method).
        A value of True indicates is a bit packed (flag_mask) number.
    return_bool : boolean
        Return a numpy array of True and False instead of 0 and 1 for
        use with subsetting a numpy array.

    Returns : numpy int array
    --------
        An array of 0 or 1 (False or True) where the test number or bit was set.

     Example
    --------
        > data = np.array([1,2,3,4])
        > mask = get_qc_test_mask(2, data)
        > mask
        array([0, 1, 1, 0])
        > mask = get_qc_test_mask(2, data, return_bool=True)
        > mask
        array([False,  True,  True, False])
        > data[mask]
        array([2, 3])
       

    """
    if bit:
        check_bit = set_bit(0, test_number) & qc_data
        tripped = np.where(check_bit > 0)
    else:
        tripped = np.where(qc_data == test_number)

    test_mask = np.zeros(qc_data.shape, dtype='int')
    # Make sure test_mask is an array. If qc_data is scalar will 
    # be retuned from np.zeros as scalar.
    test_mask = np.atleast_1d(test_mask)
    test_mask[tripped] = 1

    if return_bool:
        test_mask = np.ma.make_mask(test_mask)

    return test_mask
