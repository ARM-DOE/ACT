"""
act.qc.qctests
------------------------------

Here we define the methods for performing the tests and putting the
results in the ancillary quality control varible. If you add a test
to this file you will need to add a method reference in the main
qcfilter class definition to make it callable.

"""

import numpy as np
import pandas as pd
import xarray as xr
from act.utils import get_missing_value, convert_units


def rolling_window(data, window):
    """
    A function used by some test to efficiently calculate numpy
    statistics over a rolling window.

    Parameters
    ----------
    data : numpy array
        The data array to analyze.
    window : int
        Number of data points to perform numpy statistics over.

    Returns
    -------
        Will return a numpy array with a new dimension set to the window
        size. The numpy functions should then use -1 for dimension to
        reduce back to orginal data array size.

    Examples
    --------
    > data = np.arange(10, dtype=np.float)
    > stdev = np.nanstd(rolling_window(data, 3), axis=-1)
    > stdev
    [0.81649658 0.81649658 0.5 1. 0.5 0.81649658 0.81649658 2.1602469]

    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


# This is a Mixins class used to allow using qcfilter class that is already
# registered to the xarray object. All the methods in this class will be added
# to the qcfilter class. Doing this to make the code spread across more files
# so it is more manageable and readable. Additinal files of tests can be added
# to qcfilter by creating a new class in the new file and adding to qcfilter
# class declaration.
class QCTests:
    """
    This is a Mixins class used to allow using qcfilter class that is already
    registered to the xarray object. All the methods in this class will be added
    to the qcfilter class. Doing this to make the code spread across more files
    so it is more manageable and readable. Additinal files of tests can be added
    to qcfilter by creating a new class in the new file and adding to qcfilter
    class declaration.

    """
    def __init__(self, obj, **kwargs):

        print('test')

    def add_missing_value_test(self, var_name, missing_value=None,
                               missing_value_att_name='missing_value',
                               test_number=None, test_assessment='Bad',
                               test_meaning=None, flag_value=False,
                               prepend_text=None):
        """
        Method to add indication in quality control variable
        where data value is set to missing value.

        Parameters
        ----------
        var_name : str
            Data variable name.
        missing_value : int or float
            Optional missing value to use. If not provided will attempt
            to get it from the variable attribute or use NaN.
        missing_value_att_name : str
            Optional attribute name to use.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        if test_meaning is None:
            test_meaning = 'Value is set to missing_value.'

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        if missing_value is None:
            missing_value = get_missing_value(
                self._obj, var_name, nodefault=True)
            if (missing_value is None and
                    self._obj[var_name].values.dtype.type in
                    (type(0.0), np.float16, np.float32, np.float64)):
                missing_value = float('nan')
            else:
                missing_value = -9999

        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            missing_value = float(missing_value)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            missing_value = int(missing_value)

        if np.isnan(missing_value) is False:
            data = np.ma.masked_equal(self._obj[var_name].values,
                                      missing_value)
        else:
            data = np.ma.masked_invalid(self._obj[var_name].values)

        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        test_dict = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)

        try:
            self._obj[var_name].attrs[missing_value_att_name]
        except KeyError:
            self._obj[var_name].attrs[missing_value_att_name] = missing_value

        return test_dict

    def add_less_test(self, var_name, limit_value, test_meaning=None,
                      test_assessment='Bad', test_number=None,
                      flag_value=False, limit_attr_name=None,
                      prepend_text=None):
        """
        Method to perform a less than test (i.e. minimum value) and add
        result to ancillary quality control variable. If ancillary
        quality control variable does not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_name : str
            Optional attribute name to store the limit_value under
            quality control ancillary variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_min'
            else:
                attr_name = 'fail_min'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = ('Data value less than {}.').format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = np.ma.masked_less(self._obj[var_name].values, limit_value)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        test_dict = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)

        # Ensure min value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value = float(limit_value)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value = int(limit_value)

        self._obj[qc_var_name].attrs[attr_name] = limit_value

        return test_dict

    def add_greater_test(self, var_name, limit_value, test_meaning=None,
                         test_assessment='Bad', test_number=None,
                         flag_value=False, limit_attr_name=None,
                         prepend_text=None):
        """
        Method to perform a greater than test (i.e. maximum value) and add
        result to ancillary quality control variable. If ancillary
        quality control variable does not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_name : str
            Optional attribute name to store the limit_value under
            quality control ancillary variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_max'
            else:
                attr_name = 'fail_max'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = ('Data value greater than {}.').format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = np.ma.masked_greater(self._obj[var_name].values,
                                    limit_value)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)
        # Ensure max value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value = float(limit_value)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value = int(limit_value)

        self._obj[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_less_equal_test(self, var_name, limit_value, test_meaning=None,
                            test_assessment='Bad', test_number=None,
                            flag_value=False, limit_attr_name=None,
                            prepend_text=None):
        """
        Method to perform a less than or equal to test
        (i.e. minimum value) and add
        result to ancillary quality control variable. If ancillary
        quality control variable does not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_name : str
            Optional attribute name to store the limit_value under
            quality control ancillary variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_min'
            else:
                attr_name = 'fail_min'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = ('Data value less than '
                            'or equal to {}.').format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = np.ma.masked_less_equal(self._obj[var_name].values, limit_value)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)

        # Ensure min value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value = float(limit_value)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value = int(limit_value)

        self._obj[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_greater_equal_test(self, var_name, limit_value, test_meaning=None,
                               test_assessment='Bad', test_number=None,
                               flag_value=False, limit_attr_name=None,
                               prepend_text=None):
        """
        Method to perform a greater than or equal to test
        (i.e. maximum value) and add result to ancillary quality control
        variable. If ancillary quality control variable does not exist it
        will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_name : str
            Optional attribute name to store the limit_value under
            quality control ancillary variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_max'
            else:
                attr_name = 'fail_max'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = ('Data value greater than '
                            'or equal to {}.').format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = np.ma.masked_greater_equal(self._obj[var_name].values,
                                          limit_value)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)
        # Ensure max value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value = float(limit_value)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value = int(limit_value)

        self._obj[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_equal_to_test(self, var_name, limit_value, test_meaning=None,
                          test_assessment='Bad', test_number=None,
                          flag_value=False, limit_attr_name=None,
                          prepend_text=None):
        """
        Method to perform an equal test and add result to ancillary quality
        control variable. If ancillary quality control variable does not
        exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_name : str
            Optional attribute name to store the limit_value under
            quality control ancillary variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_equal_to'
            else:
                attr_name = 'fail_equal_to'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = 'Data value equal to {}.'.format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = np.ma.masked_equal(self._obj[var_name].values, limit_value)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)
        # Ensure max value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value = float(limit_value)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value = int(limit_value)
        elif self._obj[var_name].values.dtype.type in \
                (str(0), np.str_, np.string_):
            limit_value = str(limit_value)

        self._obj[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_not_equal_to_test(self, var_name, limit_value, test_meaning=None,
                              test_assessment='Bad', test_number=None,
                              flag_value=False, limit_attr_name=None,
                              prepend_text=None):
        """
        Method to perform a not equal to test and add result to ancillary
        quality control variable. If ancillary quality control variable does
        not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_name : str
            Optional attribute name to store the limit_value under
            quality control ancillary variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_not_equal_to'
            else:
                attr_name = 'fail_not_equal_to'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = 'Data value not equal to {}.'.format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = np.ma.masked_not_equal(self._obj[var_name].values, limit_value)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)

        # Ensure max value attribute is matching data type
        if self._obj[var_name].values.dtype in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value = float(limit_value)
        elif self._obj[var_name].values.dtype in \
                (int(0), np.int16, np.int32, np.int64):
            limit_value = int(limit_value)
        elif self._obj[var_name].values.dtype.type in \
                (str(0), np.str_, np.string_):
            limit_value = str(limit_value)

        self._obj[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_outside_test(self, var_name, limit_value_lower, limit_value_upper,
                         test_meaning=None,
                         test_assessment='Bad', test_number=None,
                         flag_value=False, limit_attr_names=None,
                         prepend_text=None):
        """
        Method to perform a less than or greater than test
        (i.e. outide minimum and maximum value) and add
        result to ancillary quality control variable. If ancillary
        quality control variable does not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value_lower : int or float
            Lower limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        limit_value_upper : int or float
            Upper limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_names : list of str
            Optional attribute name to store the limit_value under
            quality control ancillary variable. First value is
            lower limit attribute name and second value is
            upper limit attribute name.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_names is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name_lower = 'warn_lower_range'
                attr_name_upper = 'warn_upper_range'
            else:
                attr_name_lower = 'fail_lower_range'
                attr_name_upper = 'fail_upper_range'
        else:
            attr_name_lower = limit_attr_names[0]
            attr_name_upper = limit_attr_names[1]

        if test_meaning is None:
            test_meaning = ('Data value less than {} '
                            'or greater than {}.').format(attr_name_lower,
                                                          attr_name_upper)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with np.errstate(invalid='ignore'):
            data = np.ma.masked_outside(self._obj[var_name].values,
                                        limit_value_lower, limit_value_upper)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)
        # Ensure limit value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value_lower = float(limit_value_lower)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value_lower = int(limit_value_lower)

        # Ensure limit value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value_upper = float(limit_value_upper)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value_upper = int(limit_value_upper)

        self._obj[qc_var_name].attrs[attr_name_lower] = limit_value_lower

        self._obj[qc_var_name].attrs[attr_name_upper] = limit_value_upper

        return result

    def add_inside_test(self, var_name, limit_value_lower, limit_value_upper,
                        test_meaning=None, test_assessment='Bad',
                        test_number=None, flag_value=False,
                        limit_attr_names=None,
                        prepend_text=None):
        """
        Method to perform a greater than or less than test
        (i.e. between minimum and maximum value) and add
        result to ancillary quality control variable. If ancillary
        quality control variable does not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value_lower : int or float
            Lower limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        limit_value_upper : int or float
            Upper limit value to use in test. The value will be written
            to the quality control variable as an attribute.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        limit_attr_names : list of str
            Optional attribute name to store the limit_value under
            quality control ancillary variable. First value is
            lower limit attribute name and second value is
            upper limit attribute name.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        qc_var_name = self._obj.qcfilter.check_for_ancillary_qc(var_name)

        if limit_attr_names is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name_lower = 'warn_lower_range_inner'
                attr_name_upper = 'warn_upper_range_inner'
            else:
                attr_name_lower = 'fail_lower_range_inner'
                attr_name_upper = 'fail_upper_range_inner'
        else:
            attr_name_lower = limit_attr_names[0]
            attr_name_upper = limit_attr_names[1]

        if test_meaning is None:
            test_meaning = ('Data value greater than {} '
                            'or less than {}.').format(attr_name_lower,
                                                       attr_name_upper)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with np.errstate(invalid='ignore'):
            data = np.ma.masked_inside(self._obj[var_name].values,
                                       limit_value_lower, limit_value_upper)
        if data.mask.size == 1:
            data.mask = np.full(data.data.shape, data.mask, dtype=bool)
        index = np.where(data.mask)[0]

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)

        # Ensure limit value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value_lower = float(limit_value_lower)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value_lower = int(limit_value_lower)

        # Ensure limit value attribute is matching data type
        if self._obj[var_name].values.dtype.type in \
                (type(0.0), np.float16, np.float32, np.float64):
            limit_value_upper = float(limit_value_upper)
        elif self._obj[var_name].values.dtype.type in \
                (int(0), np.int8, np.int16, np.int32, np.int64):
            limit_value_upper = int(limit_value_upper)

        self._obj[qc_var_name].attrs[attr_name_lower] = limit_value_lower

        self._obj[qc_var_name].attrs[attr_name_upper] = limit_value_upper

        return result

    def add_persistence_test(self, var_name, window=10, test_limit=0.0001,
                             test_meaning=None, test_assessment='Bad',
                             test_number=None, flag_value=False,
                             prepend_text=None):
        """
        Method to perform a persistence test over 1-D data..

        Parameters
        ----------
        var_name : str
            Data variable name.
        window : int
            Optional number of data samples to use in the calculation of
            standard deviation to test for consistent data.
        test_limit : float
            Optional test limit to use where the standard deviation less
            than will trigger the test.
        test_meaning : str
            The optional text description to add to flag_meanings
            describing the test. Will add a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will set a default if not set.
        test_number : int
            Optional test number to use. If not set will ues next
            available test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        if test_meaning is None:
            test_meaning = ('Data failing persistence test. '
                            'Standard Deviation over a window of {} values '
                            'less than {}.').format(window, test_limit)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = self._obj[var_name].values

        stddev = np.nanstd(rolling_window(data, window), axis=-1)

        with np.errstate(invalid='ignore'):
            index = np.where(stddev < test_limit)

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)

        return result

    def add_difference_test(self, var_name, dataset2_dict, ds2_var_name,
                            diff_limit=None, tolerance="1m",
                            set_test_regardless=True,
                            apply_assessment_to_dataset2=None,
                            apply_tests_to_dataset2=None,
                            test_meaning=None, test_assessment='Bad',
                            test_number=None, flag_value=False,
                            prepend_text=None):
        """
        Method to perform a comparison test on time series data. Tested on 1-D
        data only.

        Parameters
        ----------
        var_name : str
            Data variable name.
        dataset2_dict : dict
            Dictionary with key equal to datastream name and value
            equal to xarray dataset containging variable to compare.
        ds2_var_name : str
            Comparison dataset variable name to compare.
        diff_limit : int or float
            Difference limit for comparison.
        apply_assessment_to_dataset2 : str or list of str
            Option to filter comparison dataset variable using corresponsing
            quality control variable using assessments. Example would be
            ['Bad'], where all quality control data with assessment Bad will
            not be used in this test.
        apply_tests_to_dataset2 : int or list of int
            Option to filter comparison dataset variable using corresponding
            quality control variable using test numbers. Example would be
            [2,4], where all quality control data with test numbers 2 or 4 set
            will not be used in this test.
        tolerance : str
            Optional text indicating the time tolerance for aligning two
            DataArrays.
        set_test_regardless : boolean
            Option to set test description even if no data in comparison data
            set.
        test_meaning : str
            Optional text description to add to flag_meanings
            describing the test. Will use a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will use a default if not set.
        test_number : int
            Optional test number to use. If not set will use next available
            test number.
        flag_value : boolean
            Indicates that the tests are stored as integers
            not bit packed values in quality control variable.
        prepend_text : str
            Optional text to prepend to the test meaning.
            Example is indicate what institution added the test.

        """
        if not isinstance(dataset2_dict, dict):
            raise ValueError('You did not provide a dictionary containing the '
                             'datastream name as the key and xarray dataset as '
                             'the value for dataset2_dict for '
                             'add_difference_test().')

        if diff_limit is None:
            raise ValueError('You did not provide a test limit for '
                             'add_difference_test().')

        datastream2 = list(dataset2_dict.keys())[0]
        dataset2 = dataset2_dict[datastream2]

        if set_test_regardless is False and type(dataset2) != xr.core.dataset.Dataset:
            return

        if test_meaning is None:
            test_meaning = ('Difference between {var1} and {ds2}:{var2} greater '
                            'than {limit} ' +
                            self._obj[var_name].attrs['units']).format(
                                var1=var_name, ds2=datastream2,
                                var2=ds2_var_name, limit=diff_limit)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        if tolerance is not None:
            tolerance = pd.Timedelta(tolerance)

        index = []
        if type(dataset2) == xr.core.dataset.Dataset:
            if apply_assessment_to_dataset2 is not None or apply_tests_to_dataset2 is not None:
                dataset2[ds2_var_name].values = dataset2.qcfilter.get_masked_data(
                    ds2_var_name, rm_assessments=apply_assessment_to_dataset2,
                    rm_tests=apply_tests_to_dataset2, return_nan_array=True)

            df_a = pd.DataFrame({'time': self._obj['time'].values,
                                 var_name: self._obj[var_name].values})
            data_b = convert_units(dataset2[ds2_var_name].values,
                                   dataset2[ds2_var_name].attrs['units'],
                                   self._obj[var_name].attrs['units'])
            ds2_var_name = ds2_var_name + '_newname'
            df_b = pd.DataFrame({'time': dataset2['time'].values,
                                 ds2_var_name: data_b})

            if tolerance is not None:
                tolerance = pd.Timedelta(tolerance)

            pd_c = pd.merge_asof(df_a, df_b, on='time', tolerance=tolerance,
                                 direction="nearest")

            with np.errstate(invalid='ignore'):
                diff = np.absolute(pd_c[var_name] - pd_c[ds2_var_name])
                index = np.where(diff > diff_limit)

        result = self._obj.qcfilter.add_test(
            var_name, index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value)

        return result
