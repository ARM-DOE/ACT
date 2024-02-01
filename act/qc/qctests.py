"""
Here we define the methods for performing the tests and putting the
results in the ancillary quality control varible. If you add a test
to this file you will need to add a method reference in the main
qcfilter class definition to make it callable.

"""

import warnings

import dask.array as da
from metpy.units import units
from metpy.calc import add_height_to_pressure
import numpy as np
import pandas as pd
import xarray as xr

from act.utils.data_utils import convert_units, get_missing_value


# This is a Mixins class used to allow using qcfilter class that is already
# registered to the Xarray dataset. All the methods in this class will be added
# to the qcfilter class. Doing this to make the code spread across more files
# so it is more manageable and readable. Additinal files of tests can be added
# to qcfilter by creating a new class in the new file and adding to qcfilter
# class declaration.
class QCTests:
    """
    This is a Mixins class used to allow using qcfilter class that is already
    registered to the Xarray dataset. All the methods in this class will be added
    to the qcfilter class. Doing this to make the code spread across more files
    so it is more manageable and readable. Additinal files of tests can be added
    to qcfilter by creating a new class in the new file and adding to qcfilter
    class declaration.

    """

    def __init__(self, ds, **kwargs):
        self._ds = ds

    def add_missing_value_test(
        self,
        var_name,
        missing_value=None,
        missing_value_att_name='missing_value',
        test_number=None,
        test_assessment='Bad',
        test_meaning=None,
        flag_value=False,
        prepend_text=None,
        use_dask=False,
    ):
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        if test_meaning is None:
            test_meaning = 'Value is set to missing_value.'

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        if missing_value is None:
            missing_value = get_missing_value(self._ds, var_name, nodefault=True)
            if missing_value is None and self._ds[var_name].values.dtype.type in (
                float,
                np.float16,
                np.float32,
                np.float64,
            ):
                missing_value = float('nan')
            else:
                missing_value = -9999

        # Ensure missing_value attribute is matching data type
        missing_value = np.array(missing_value, dtype=self._ds[var_name].values.dtype.type)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                if np.isnan(missing_value) is False:
                    index = da.where(
                        self._ds[var_name].data == missing_value, True, False
                    ).compute()
                else:
                    index = da.isnan(self._ds[var_name].data).compute()
            else:
                if np.isnan(missing_value) is False:
                    index = np.equal(self._ds[var_name].values, missing_value)
                else:
                    index = np.isnan(self._ds[var_name].values)

        test_dict = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        try:
            self._ds[var_name].attrs[missing_value_att_name]
        except KeyError:
            self._ds[var_name].attrs[missing_value_att_name] = missing_value

        return test_dict

    def add_less_test(
        self,
        var_name,
        limit_value,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_name=None,
        prepend_text=None,
        use_dask=False,
    ):
        """
        Method to perform a less than test (i.e. minimum value) and add
        result to ancillary quality control variable. If ancillary
        quality control variable does not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float or None
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute. If set
            to None, will return without adding test.
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        if limit_value is None:
            return

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

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index = da.where(self._ds[var_name].data < limit_value, True, False).compute()
            else:
                index = np.less(self._ds[var_name].values, limit_value)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value = np.array(limit_value, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_greater_test(
        self,
        var_name,
        limit_value,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_name=None,
        prepend_text=None,
        use_dask=False,
    ):
        """
        Method to perform a greater than test (i.e. maximum value) and add
        result to ancillary quality control variable. If ancillary
        quality control variable does not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float or None
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute. If set
            to None will return without setting test.
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        if limit_value is None:
            return

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

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index = da.where(self._ds[var_name].data > limit_value, True, False).compute()
            else:
                index = np.greater(self._ds[var_name].values, limit_value)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value = np.array(limit_value, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_less_equal_test(
        self,
        var_name,
        limit_value,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_name=None,
        prepend_text=None,
        use_dask=False,
    ):
        """
        Method to perform a less than or equal to test
        (i.e. minimum value) and add result to ancillary quality control
        variable. If ancillary quality control variable does not exist it
        will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float or None
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute. If set
            to None will return without setttin test.
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        if limit_value is None:
            return

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_min'
            else:
                attr_name = 'fail_min'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = ('Data value less than ' 'or equal to {}.').format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            index = np.less_equal(self._ds[var_name].values, limit_value)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index = da.where(self._ds[var_name].data <= limit_value, True, False).compute()
            else:
                index = np.less_equal(self._ds[var_name].values, limit_value)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value = np.array(limit_value, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_greater_equal_test(
        self,
        var_name,
        limit_value,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_name=None,
        prepend_text=None,
        use_dask=False,
    ):
        """
        Method to perform a greater than or equal to test
        (i.e. maximum value) and add result to ancillary quality control
        variable. If ancillary quality control variable does not exist it
        will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float or None
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute. If set
            to None will return without setttin test.
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        if limit_value is None:
            return

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_max'
            else:
                attr_name = 'fail_max'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = ('Data value greater than ' 'or equal to {}.').format(attr_name)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index = da.where(self._ds[var_name].data >= limit_value, True, False).compute()
            else:
                index = np.greater_equal(self._ds[var_name].values, limit_value)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value = np.array(limit_value, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_equal_to_test(
        self,
        var_name,
        limit_value,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_name=None,
        prepend_text=None,
        use_dask=False,
    ):
        """
        Method to perform an equal test and add result to ancillary quality
        control variable. If ancillary quality control variable does not
        exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float or None
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute. If set
            to None will return without setttin test.
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        if limit_value is None:
            return

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_equal_to'
            else:
                attr_name = 'fail_equal_to'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = f'Data value equal to {attr_name}.'

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index = da.where(self._ds[var_name].data == limit_value, True, False).compute()
            else:
                index = np.equal(self._ds[var_name].values, limit_value)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value = np.array(limit_value, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_not_equal_to_test(
        self,
        var_name,
        limit_value,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_name=None,
        prepend_text=None,
        use_dask=False,
    ):
        """
        Method to perform a not equal to test and add result to ancillary
        quality control variable. If ancillary quality control variable does
        not exist it will be created.

        Parameters
        ----------
        var_name : str
            Data variable name.
        limit_value : int or float or None
            Limit value to use in test. The value will be written
            to the quality control variable as an attribute. If set
            to None will return without setttin test.
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        if limit_value is None:
            return

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_not_equal_to'
            else:
                attr_name = 'fail_not_equal_to'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = f'Data value not equal to {attr_name}.'

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index = da.where(self._ds[var_name].data != limit_value, True, False).compute()
            else:
                index = np.not_equal(self._ds[var_name].values, limit_value)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value = np.array(limit_value, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name] = limit_value

        return result

    def add_outside_test(
        self,
        var_name,
        limit_value_lower,
        limit_value_upper,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_names=None,
        prepend_text=None,
        use_dask=False,
    ):
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """

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
            test_meaning = ('Data value less than {} ' 'or greater than {}.').format(
                attr_name_lower, attr_name_upper
            )

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index1 = da.where(self._ds[var_name].data < limit_value_lower, True, False)
                index2 = da.where(self._ds[var_name].data > limit_value_upper, True, False)
                index = (index1 | index2).compute()
            else:
                data = np.ma.masked_outside(
                    self._ds[var_name].values, limit_value_lower, limit_value_upper
                )
                if data.mask.size == 1:
                    data.mask = np.full(data.data.shape, data.mask, dtype=bool)

                index = data.mask

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value_lower = np.array(limit_value_lower, dtype=self._ds[var_name].values.dtype.type)
        limit_value_upper = np.array(limit_value_upper, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name_lower] = limit_value_lower
        self._ds[qc_var_name].attrs[attr_name_upper] = limit_value_upper

        return result

    def add_inside_test(
        self,
        var_name,
        limit_value_lower,
        limit_value_upper,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        limit_attr_names=None,
        prepend_text=None,
        use_dask=False,
    ):
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
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """

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
            test_meaning = ('Data value greater than {} ' 'or less than {}.').format(
                attr_name_lower, attr_name_upper
            )

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index1 = da.where(self._ds[var_name].data > limit_value_lower, True, False)
                index2 = da.where(self._ds[var_name].data < limit_value_upper, True, False)
                index = (index1 & index2).compute()
            else:
                data = np.ma.masked_inside(
                    self._ds[var_name].values, limit_value_lower, limit_value_upper
                )
                if data.mask.size == 1:
                    data.mask = np.full(data.data.shape, data.mask, dtype=bool)

                index = data.mask

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure limit_value attribute is matching data type
        limit_value_lower = np.array(limit_value_lower, dtype=self._ds[var_name].values.dtype.type)
        limit_value_upper = np.array(limit_value_upper, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name_lower] = limit_value_lower
        self._ds[qc_var_name].attrs[attr_name_upper] = limit_value_upper

        return result

    def add_persistence_test(
        self,
        var_name,
        window=10,
        test_limit=0.0001,
        min_periods=1,
        center=True,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        prepend_text=None,
    ):
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
        min_periods : int
            Optional number of minimum values to use in the moving window.
            Setting to 1 so this correctly handles NaNs.
        center : boolean
            Optional where within the moving window to report the standard
            deviation values. Used in the .rolling.std() calculation with xarray.
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

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        data = self._ds[var_name]
        if window > data.size:
            window = data.size

        if test_meaning is None:
            test_meaning = (
                'Data failing persistence test. '
                'Standard Deviation over a window of {} values '
                'less than {}.'
            ).format(window, test_limit)

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            stddev = data.rolling(time=window, min_periods=min_periods, center=True).std()
            index = stddev < test_limit

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        return result

    def add_difference_test(
        self,
        var_name,
        dataset2_dict=None,
        ds2_var_name=None,
        diff_limit=None,
        tolerance='1m',
        set_test_regardless=True,
        apply_assessment_to_dataset2=None,
        apply_tests_to_dataset2=None,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        prepend_text=None,
    ):
        """
        Method to perform a comparison test on time series data. Tested on 1-D
        data only. Will check if units and long_name indicate a direction and
        compensate for 0 to 360 degree transition.

        Parameters
        ----------
        var_name : str
            Data variable name.
        dataset2_dict : dict
            Dictionary with key equal to datastream name and value
            equal to xarray dataset containging variable to compare. If no provided
            will assume second dataset is the same as self dataset.
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

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """

        if set_test_regardless is False and dataset2_dict is None:
            return

        if dataset2_dict is None:
            dataset2_dict = {'second_dataset': self._ds}

        else:
            if not isinstance(dataset2_dict, dict):
                raise ValueError(
                    'You did not provide a dictionary containing the '
                    'datastream name as the key and xarray dataset as '
                    'the value for dataset2_dict for add_difference_test().'
                )

            if diff_limit is None:
                raise ValueError('You did not provide a test limit for add_difference_test().')

        datastream2 = list(dataset2_dict.keys())[0]
        dataset2 = dataset2_dict[datastream2]

        if test_meaning is None:
            if dataset2 is self._ds:
                var_name2 = f'{ds2_var_name}'
            else:
                var_name2 = f'{datastream2}:{ds2_var_name}'

            test_meaning = (
                f'Difference between {var_name} and {var_name2} '
                f'greater than {diff_limit} {self._ds[var_name].attrs["units"]}'
            )

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        if tolerance is not None:
            tolerance = pd.Timedelta(tolerance)

        index = []
        if isinstance(dataset2, xr.core.dataset.Dataset):
            if apply_assessment_to_dataset2 is not None or apply_tests_to_dataset2 is not None:
                dataset2[ds2_var_name].values = dataset2.qcfilter.get_masked_data(
                    ds2_var_name,
                    rm_assessments=apply_assessment_to_dataset2,
                    rm_tests=apply_tests_to_dataset2,
                    return_nan_array=True,
                )

            df_a = pd.DataFrame(
                {'time': self._ds['time'].values, var_name: self._ds[var_name].values}
            )
            data_b = convert_units(
                dataset2[ds2_var_name].values,
                dataset2[ds2_var_name].attrs['units'],
                self._ds[var_name].attrs['units'],
            )
            ds2_var_name = ds2_var_name + '_newname'
            df_b = pd.DataFrame({'time': dataset2['time'].values, ds2_var_name: data_b})

            if tolerance is not None:
                tolerance = pd.Timedelta(tolerance)

            pd_c = pd.merge_asof(df_a, df_b, on='time', tolerance=tolerance, direction='nearest')

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                # Check if variable is for wind direction comparisons. Fix
                # for 0 - 360 degrees transition. This is done by adding 360 degrees to
                # all wind values and using modulus to get the minimum difference number.
                # This is done for both a-b and b-a and then choosing the minimum number
                # to compensate for large differences.
                wdir_units = ['deg', 'degree', 'degrees', 'degs']
                if (
                    self._ds[var_name].attrs['units'] in wdir_units
                    and 'direction' in self._ds[var_name].attrs['long_name'].lower()
                ):
                    diff1 = np.mod(
                        np.absolute((pd_c[var_name] + 360.0) - (pd_c[ds2_var_name] + 360.0)),
                        360,
                    )
                    diff2 = np.mod(
                        np.absolute((pd_c[ds2_var_name] + 360.0) - (pd_c[var_name] + 360.0)),
                        360,
                    )
                    diff = np.array([diff1, diff2])
                    diff = np.nanmin(diff, axis=0)

                else:
                    diff = np.absolute(pd_c[var_name] - pd_c[ds2_var_name])

                index = diff > diff_limit

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        return result

    def add_delta_test(
        self,
        var_name,
        diff_limit=1,
        test_meaning=None,
        limit_attr_name=None,
        test_assessment='Indeterminate',
        test_number=None,
        flag_value=False,
        prepend_text=None,
    ):
        """
        Method to perform a difference test on adjacent values in time series.
        Will flag both values where a difference is greater
        than or equal to the difference limit. Tested with 1-D data only. Not
        sure what will happen with higher dimentioned data.

        Parameters
        ----------
        var_name : str
            Data variable name.
        diff_limit : int or float
            Difference limit
        test_meaning : str
            Optional text description to add to flag_meanings
            describing the test. Will use a default if not set.
        limit_attr_name : str
            Optional attribute name to store the limit_value under
            quality control ancillary variable.
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

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """

        if limit_attr_name is None:
            if test_assessment == 'Suspect' or test_assessment == 'Indeterminate':
                attr_name = 'warn_delta'
            else:
                attr_name = 'fail_delta'
        else:
            attr_name = limit_attr_name

        if test_meaning is None:
            test_meaning = f'Difference between current and previous values exceeds {attr_name}.'

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            # Check if variable is for wind direction comparisons by units. Fix
            # for 0 - 360 degrees transition. This is done by adding 360 degrees to
            # all wind values and using modulus to get the minimum difference number.
            wdir_units = ['deg', 'degree', 'degrees', 'degs']
            if (
                self._ds[var_name].attrs['units'] in wdir_units
                and 'direction' in self._ds[var_name].attrs['long_name'].lower()
            ):
                abs_diff = np.mod(np.abs(np.diff(self._ds[var_name].values)), 360)
            else:
                abs_diff = np.abs(np.diff(self._ds[var_name].values))

            index = np.where(abs_diff >= diff_limit)[0]
            if index.size > 0:
                index = np.append(index, index + 1)
                index = np.unique(index)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        # Ensure min value attribute is matching data type
        diff_limit = np.array(diff_limit, dtype=self._ds[var_name].values.dtype.type)

        qc_var_name = result['qc_variable_name']
        self._ds[qc_var_name].attrs[attr_name] = diff_limit

        return result

    def add_iqr_test(
        self,
        var_name,
        coef=1.5,
        test_meaning=None,
        test_assessment='Indeterminate',
        test_number=None,
        flag_value=False,
        prepend_text=None,
    ):
        """
        Method to perform an interquartile range outliers test on 1D data.
        Data that lie within the lower and upper limits are considered
        non-outliers. The lower limit is the number that lies coef IQRs below
        the first quartile; the upper limit is the number that lies coef IQRs
        above the third quartile. This method will flag data
        failing the test in the corresponding quality control variable.

        The library used to perform test does not accept NaN values. The NaN
        values will be filtered out prior to testing and outlier values will
        be matched after. This can cause the test to run slower on large data
        sets.

        Parameters
        ----------
        var_name : str
            Data variable name.
        coef : float
            Coefficient by which interquartile range is multiplied.
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

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc
            variable name, test_number, test_meaning, test_assessment

        """

        try:
            from scikit_posthocs import outliers_iqr
        except ImportError:
            raise ImportError(
                'scikit_posthocs needs to be installed on your system to ' 'run add_iqr_test.'
            )

        if test_meaning is None:
            test_meaning = (
                'Value outside of interquartile range test range with ' f'a coefficient of {coef}'
            )

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = self._ds[var_name].values
        data = data[np.isfinite(data)]

        fail_data = outliers_iqr(data, ret='outliers')

        index = []
        if fail_data.size > 0:
            index = np.array([], dtype=int)
            for ii in np.unique(fail_data):
                ind = np.atleast_1d(self._ds[var_name].values == ii).nonzero()
                index = np.append(index, ind)

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        return result

    def add_gesd_test(
        self,
        var_name,
        outliers=5,
        alpha=0.05,
        test_meaning=None,
        test_assessment='Indeterminate',
        test_number=None,
        flag_value=False,
        prepend_text=None,
    ):
        """
        Method to perform generalized Extreme Studentized Deviate test to
        detect one or more outliers in a univariate data set that follows an
        approximately normal distribution. Default is to find 5 outliers but
        can overestimate number of outliers and will only flag values
        determined to be outliers. If set to find one outlier is the Grubbs
        test.

        The library used to perform test does not accept NaN values. The NaN
        values will be filtered out prior to testing and outlier values will
        be matched after. This can cause the test to run slower on large data
        sets.

        Parameters
        ----------
        var_name : str
            Data variable name.
        outliers : int or float
            Number of outliers to test for. If set to 1 is the Grubbs test.
            If set to float values less than one will calcualte the number of
            outliers to test for. Float value from 0 to 0.9 will be multiplied
            by the number of data values to determine number of outliers to
            check. If set to value larger than 0.9 will use 0.9.
        alpha : float
            Significance level for a hypothesis test
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

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc
            variable name, test_number, test_meaning, test_assessment

        """

        try:
            from scikit_posthocs import outliers_gesd
        except ImportError:
            raise ImportError(
                'scikit_posthocs needs to be installed on your system to ' 'run add_gesd_test.'
            )

        if test_meaning is None:
            test_meaning = (
                'Value failed generalized Extreme Studentized Deviate test '
                f'with an alpha of {alpha}'
            )

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        data = self._ds[var_name].values

        if outliers < 1:
            if outliers > 0.9:
                outliers = 0.9
            outliers = int(np.ceil(outliers * data.size))
        else:
            outliers = int(outliers)

        data = data[np.isfinite(data)]

        index = outliers_gesd(data, outliers=outliers, hypo=True, alpha=alpha)

        if index.dtype == np.bool_:
            fail_data = data[index]
            index = np.array([], dtype=int)
            for ii in np.unique(fail_data):
                ind = (self._ds[var_name].values == ii).nonzero()
                index = np.append(index, ind)
        else:
            index = []

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        return result

    def add_atmospheric_pressure_test(
        self,
        var_name,
        alt_name='alt',
        test_limit=3.5,
        sea_level_pressure=101.325,
        bias=0.35,
        test_meaning=None,
        test_assessment='Bad',
        test_number=None,
        flag_value=False,
        prepend_text=None,
        use_dask=False,
    ):
        """
        Method to perform a limit test on atmospheric pressure data using
        pressure derived from altitude value. Will use the derived pressure as
        a mean and apply upper and lower limit test on the data, flagging
        where outside the limit range.

        Parameters
        ----------
        var_name : str
            Data variable name within the Dataset
        alt_name : str
            Altitude data variable name within the Dataset
        test_limit : int or float
            Test range value to add/subtract from derived pressure value to
            set range limits.
        sea_level_pressure : float
            Sea level pressure in kPa used for deriving pressure at altitude.
        bias : float
            The derived pressure value in has a slight bias from typically
            measured values. This allows adjusting the limits. This value in
            units of kPa is added to the derived pressure value.
        test_meaning : str
            Optional text description to add to flag_meanings describing the
            test. Will use a default if not set.
        test_assessment : str
            Optional single word describing the assessment of the test.
            Will use a default if not set.
        test_number : int
            Optional test number to use. If not set will use next available
            test number.
        flag_value : boolean
            Indicates that the tests are stored as integers not bit packed
            values in quality control variable.
        prepend_text : str
            Optional text to prepend to the test meaning. Example is indicate
            what institution added the test.
        use_dask : boolean
            Option to use Dask for searching if data is stored in a Dask array.

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc
            variable name, test_number, test_meaning, test_assessment.

        Examples
        --------
            .. code-block:: python

                result = ds.qcfilter.add_atmospheric_pressure_test('atmos_pressure', use_dask=True)
                print(result)

                {'test_number': 1, 'test_meaning': 'Value outside of atmospheric pressure range test range: 94.41 to 101.41 kPa',
                 'test_assessment': 'Bad', 'qc_variable_name': 'qc_atmos_pressure', 'variable_name': 'atmos_pressure'}


        """
        data_units = self._ds[var_name].attrs['units']
        working_units = 'kPa'
        test_limit = test_limit * units(working_units)
        test_limit = test_limit.to(data_units)
        altitude = self._ds[alt_name].values
        if altitude.size > 1:
            altitude = np.nanmean(altitude)

        altitude = altitude * units(self._ds[alt_name].attrs['units'])

        sea_level_pressure = sea_level_pressure * units(working_units)
        bias = bias * units(working_units)
        bias = bias.to(data_units)
        pressure = add_height_to_pressure(sea_level_pressure, altitude)
        pressure = pressure.to(data_units)
        pressure += bias

        lower_limit = pressure - test_limit
        upper_limit = pressure + test_limit
        lower_limit = lower_limit.magnitude
        upper_limit = upper_limit.magnitude

        if test_meaning is None:
            test_meaning = (
                'Value outside of atmospheric pressure range test range: '
                f'{round(lower_limit, 2)} to {round(upper_limit, 2)} {data_units}'
            )

        if prepend_text is not None:
            test_meaning = ': '.join((prepend_text, test_meaning))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            if use_dask and isinstance(self._ds[var_name].data, da.Array):
                index1 = da.where(self._ds[var_name].data < lower_limit, True, False)
                index2 = da.where(self._ds[var_name].data > upper_limit, True, False)
                index = (index1 | index2).compute()
            else:
                index = (self._ds[var_name].values > upper_limit) | (
                    self._ds[var_name].values < lower_limit
                )

        result = self._ds.qcfilter.add_test(
            var_name,
            index=index,
            test_number=test_number,
            test_meaning=test_meaning,
            test_assessment=test_assessment,
            flag_value=flag_value,
        )

        return result
