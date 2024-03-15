"""
Functions and methods for creating ancillary quality control variables
and filters (masks) which can be used with various corrections
routines in ACT.

"""

import dask
import numpy as np
import xarray as xr

from act.qc import comparison_tests, qctests, bsrn_tests


@xr.register_dataset_accessor('qcfilter')
class QCFilter(qctests.QCTests, comparison_tests.QCTests, bsrn_tests.QCTests):
    """
    A class for building quality control variables containing arrays for
    filtering data based on a set of test condition typically based on the
    values in the data fields. These filters can be used in various
    algorithms and calculations within ACT.

    """

    def __init__(self, ds):
        """initialize"""
        self._ds = ds

    def check_for_ancillary_qc(self, var_name, add_if_missing=True, cleanup=False, flag_type=False):
        """
        Method to check if a quality control variable exist in the dataset
        and return the quality control varible name.
        Will call create_qc_variable() to make variable if does not exist
        and update_ancillary_variable() to ensure linkage between data and
        quality control variable. Can also be used just to get the
        corresponding quality control variable name with adding if
        it is missing.

        Parameters
        ----------
        var_name : str
            Data variable name.
        add_if_missing : boolean
            Add quality control variable if missing from teh dataset. Will raise
            and exception if the var_name does not exist in Dataset. Set to False
            to not raise exception.
        cleanup : boolean
            Option to run qc.clean.cleanup() method on the dataset
            to ensure the dataset was updated from ARM QC to the
            correct standardized QC.
        flag_type : boolean
            Indicating the QC variable uses flag_values instead of
            flag_masks.

        Returns
        -------
        qc_var_name : str or None
            Name of existing or new quality control variable. Returns
            None if no existing quality control variable is found and
            add_if_missing is set to False.

        Examples
        --------
            .. code-block:: python

                from act.tests import EXAMPLE_METE40
                from act.io.arm import read_arm_netcdf
                ds = read_arm_netcdf(EXAMPLE_METE40, cleanup_qc=True)
                qc_var_name = ds.qcfilter.check_for_ancillary_qc('atmos_pressure')
                print(f'qc_var_name: {qc_var_name}')
                qc_var_name = ds.qcfilter.check_for_ancillary_qc('the_greatest_variable_ever',
                    add_if_missing=False)
                print(f'qc_var_name: {qc_var_name}')

        """
        qc_var_name = None
        try:
            ancillary_variables = self._ds[var_name].attrs['ancillary_variables']
            if isinstance(ancillary_variables, str):
                ancillary_variables = ancillary_variables.split()

            for var in ancillary_variables:
                for attr, value in self._ds[var].attrs.items():
                    if attr == 'standard_name' and 'quality_flag' in value:
                        qc_var_name = var

            if add_if_missing and qc_var_name is None:
                qc_var_name = self._ds.qcfilter.create_qc_variable(var_name, flag_type=flag_type)

        except KeyError:
            # Since no ancillary_variables exist look for ARM style of QC
            # variable name. If it exists use it else create new
            # QC varaible.
            if add_if_missing:
                try:
                    self._ds['qc_' + var_name]
                    qc_var_name = 'qc_' + var_name
                except KeyError:
                    qc_var_name = self._ds.qcfilter.create_qc_variable(
                        var_name, flag_type=flag_type
                    )

        # Make sure data varaible has a variable attribute linking
        # data variable to QC variable.
        if add_if_missing:
            self._ds.qcfilter.update_ancillary_variable(var_name, qc_var_name)

        # Clean up quality control variables to the requried standard in the
        # xarray dataset.
        if cleanup:
            self._ds.clean.cleanup(handle_missing_value=True, link_qc_variables=False)

        return qc_var_name

    def create_qc_variable(
        self, var_name, flag_type=False, flag_values_set_value=0, qc_var_name=None
    ):
        """
        Method to create a quality control variable in the dataset.
        Will try not to destroy the qc variable by appending numbers
        to the variable name if needed.

        Parameters
        ----------
        var_name : str
            Data variable name.
        flag_type : boolean
            If an integer flag type should be created instead of
            bitpacked mask type. Will create flag_values instead of
            flag_masks.
        flag_values_set_value : int
            Initial flag value to use when initializing array.
        qc_var_name : str
            Optional new quality control variable name. If not set
            will create one using \\"qc\\_\\" prepended to the data
            variable name. If the name given or created is taken
            will append a number that does not have a conflict.

        Returns
        -------
        qc_var_name : str
            Name of new quality control variable created.

        Examples
        --------
            .. code-block:: python

                from act.tests import EXAMPLE_AOSMET
                from act.io.arm import read_arm_netcdf
                ds = read_arm_netcdf(EXAMPLE_AOSMET)
                qc_var_name = ds.qcfilter.create_qc_variable('temperature_ambient')
                print(qc_var_name)
                print(ds[qc_var_name])

        """

        # Make QC variable long name. The variable long_name attribute
        # may not exist so catch that error and set to default.
        try:
            qc_variable_long_name = (
                'Quality check results on field: ' + self._ds[var_name].attrs['long_name']
            )
        except KeyError:
            qc_variable_long_name = 'Quality check results for ' + var_name

        # Make a new quality control variable name. Check if exists in the
        # dataset. If so loop through creation of new name until one is
        # found that will not replace existing variable.
        if qc_var_name is None:
            qc_var_name = 'qc_' + var_name

        variable_names = list(self._ds.data_vars)
        if qc_var_name in variable_names:
            for ii in range(1, 100):
                temp_qc_var_name = '_'.join([qc_var_name, str(ii)])
                if temp_qc_var_name not in variable_names:
                    qc_var_name = temp_qc_var_name
                    break

        # Create the QC variable filled with 0 values matching the
        # shape of data variable.
        try:
            qc_data = dask.array.from_array(
                np.zeros_like(self._ds[var_name].values, dtype=np.int32),
                chunks=self._ds[var_name].data.chunksize,
            )
        except AttributeError:
            qc_data = np.zeros_like(self._ds[var_name].values, dtype=np.int32)

        # Updating to use coords instead of dim, which caused a loss of
        # attribuets as noted in Issue 347
        self._ds[qc_var_name] = xr.DataArray(
            data=qc_data,
            coords=self._ds[var_name].coords,
            attrs={'long_name': qc_variable_long_name, 'units': '1'},
        )

        # Update if using flag_values and don't want 0 to be default value.
        if flag_type and flag_values_set_value != 0:
            self._ds[qc_var_name].values = self._ds[qc_var_name].values + int(flag_values_set_value)

        # Add requried variable attributes.
        if flag_type:
            self._ds[qc_var_name].attrs['flag_values'] = []
        else:
            self._ds[qc_var_name].attrs['flag_masks'] = []
        self._ds[qc_var_name].attrs['flag_meanings'] = []
        self._ds[qc_var_name].attrs['flag_assessments'] = []
        self._ds[qc_var_name].attrs['standard_name'] = 'quality_flag'

        self.update_ancillary_variable(var_name, qc_var_name=qc_var_name)

        return qc_var_name

    def update_ancillary_variable(self, var_name, qc_var_name=None):
        """
        Method to check if ancillary_variables variable attribute
        is set with quality control variable name.

        Parameters
        ----------
        var_name : str
            Data variable name.
        qc_var_name : str
            quality control variable name. If not given will attempt
            to get the name from data variable ancillary_variables
            attribute.

        Examples
        --------
            .. code-block:: python

                from act.tests import EXAMPLE_AOSMET
                from act.io.arm import read_arm_netcdf
                ds = read_arm_netcdf(EXAMPLE_AOSMET)
                var_name = 'temperature_ambient'
                qc_var_name = ds.qcfilter.create_qc_variable(var_name)
                del ds[var_name].attrs['ancillary_variables']
                ds.qcfilter.update_ancillary_variable(var_name, qc_var_name)
                print(ds[var_name].attrs['ancillary_variables'])

        """
        if qc_var_name is None:
            qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(var_name, add_if_missing=False)

        if qc_var_name is None:
            return

        try:
            ancillary_variables = self._ds[var_name].attrs['ancillary_variables']
            if qc_var_name not in ancillary_variables:
                ancillary_variables = ' '.join([ancillary_variables, qc_var_name])
        except KeyError:
            ancillary_variables = qc_var_name

        self._ds[var_name].attrs['ancillary_variables'] = ancillary_variables

    def add_test(
        self,
        var_name,
        index=None,
        test_number=None,
        test_meaning=None,
        test_assessment='Bad',
        flag_value=False,
        recycle=False,
    ):
        """
        Method to add a new test/filter to a quality control variable.

        Parameters
        ----------
        var_name : str
            data variable name
        index : int, bool, list of int or bool, numpy array, tuple of numpy arrays, None
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
            update to be capitalized.
        flag_value : boolean
            Switch to use flag_values integer quality control.
        recyle : boolean
            Option to use number less than next highest test if available. For example
            tests 1, 2, 4, 5 are set. Set to true the next test chosen will be 3, else
            will be 6.

        Returns
        -------
        test_dict : dict
            A dictionary containing information added to the QC
            variable.

        Examples
        --------
            .. code-block:: python

                result = ds.qcfilter.add_test(var_name, test_meaning='Birds!')

        """
        test_dict = {}

        if test_meaning is None:
            raise ValueError(
                'You need to provide a value for test_meaning '
                'keyword when calling the add_test method'
            )

        # This ensures the indexing will work even if given float values.
        # Preserves tuples from np.where() or boolean arrays for standard
        # python indexing.
        if index is not None and not isinstance(index, (np.ndarray, tuple)):
            index = np.array(index)
            if index.dtype.kind not in np.typecodes['AllInteger']:
                index = index.astype(int)

        # Ensure assessment is capitalized to be consistent
        test_assessment = test_assessment.capitalize()

        qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(var_name, flag_type=flag_value)

        if test_number is None:
            test_number = self._ds.qcfilter.available_bit(qc_var_name, recycle=recycle)

        self._ds.qcfilter.set_test(var_name, index, test_number, flag_value)

        if flag_value:
            try:
                self._ds[qc_var_name].attrs['flag_values'].append(test_number)
            except KeyError:
                self._ds[qc_var_name].attrs['flag_values'] = [test_number]
        else:
            # Determine if flag_masks test number is too large for current data type.
            # If so up convert data type.
            flag_masks = np.array(self._ds[qc_var_name].attrs['flag_masks'])
            mask_dtype = flag_masks.dtype
            if not np.issubdtype(mask_dtype, np.integer):
                mask_dtype = np.uint32

            if np.iinfo(mask_dtype).max - set_bit(0, test_number) <= -1:
                if mask_dtype == np.int8 or mask_dtype == np.uint8:
                    mask_dtype = np.uint16
                elif mask_dtype == np.int16 or mask_dtype == np.uint16:
                    mask_dtype = np.uint32
                elif mask_dtype == np.int32 or mask_dtype == np.uint32:
                    mask_dtype = np.uint64

            flag_masks = flag_masks.astype(mask_dtype)
            flag_masks = np.append(flag_masks, np.array(set_bit(0, test_number), dtype=mask_dtype))
            self._ds[qc_var_name].attrs['flag_masks'] = list(flag_masks)

        try:
            self._ds[qc_var_name].attrs['flag_meanings'].append(test_meaning)
        except KeyError:
            self._ds[qc_var_name].attrs['flag_meanings'] = [test_meaning]

        try:
            self._ds[qc_var_name].attrs['flag_assessments'].append(test_assessment)
        except KeyError:
            self._ds[qc_var_name].attrs['flag_assessments'] = [test_assessment]

        test_dict['test_number'] = test_number
        test_dict['test_meaning'] = test_meaning
        test_dict['test_assessment'] = test_assessment
        test_dict['qc_variable_name'] = qc_var_name
        test_dict['variable_name'] = var_name

        return test_dict

    def remove_test(
        self,
        var_name=None,
        test_number=None,
        qc_var_name=None,
        flag_value=False,
        flag_values_reset_value=0,
    ):
        """
        Method to remove a test/filter from a quality control variable. Must set
        var_name or qc_var_name.

        Parameters
        ----------
        var_name : str or None
            Data variable name.
        test_number : int
            Test number to remove.
        qc_var_name : str or None
            Quality control variable name. Ignored if var_name is set.
        flag_value : boolean
            Switch to use flag_values integer quality control.
        flag_values_reset_value : int
            Value to use when resetting a flag_values value to not be set.

        Examples
        --------
            .. code-block:: python

                ds.qcfilter.remove_test(var_name, test_number=3)

        """
        if test_number is None:
            raise ValueError(
                'You need to provide a value for test_number '
                'keyword when calling the remove_test() method'
            )

        if var_name is None and qc_var_name is None:
            raise ValueError(
                'You need to provide a value for var_name or qc_var_name '
                'keyword when calling the remove_test() method'
            )

        if var_name is not None:
            qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(var_name)

        # Determine which index is using the test number
        index = None
        if flag_value:
            flag_values = self._ds[qc_var_name].attrs['flag_values']
            for ii, flag_num in enumerate(flag_values):
                if flag_num == test_number:
                    index = ii
                    break
        else:
            flag_masks = self._ds[qc_var_name].attrs['flag_masks']
            for ii, bit_num in enumerate(flag_masks):
                if parse_bit(bit_num)[0] == test_number:
                    index = ii
                    break

        # If can't find the index of test return before doing anything.
        if index is None:
            return

        if flag_value:
            remove_index = self._ds.qcfilter.get_qc_test_mask(
                var_name=var_name,
                qc_var_name=qc_var_name,
                test_number=test_number,
                return_index=True,
                flag_value=True,
            )
            self._ds.qcfilter.unset_test(
                var_name=var_name,
                qc_var_name=qc_var_name,
                index=remove_index,
                test_number=test_number,
                flag_value=flag_value,
                flag_values_reset_value=flag_values_reset_value,
            )
            del flag_values[index]
            self._ds[qc_var_name].attrs['flag_values'] = flag_values

        else:
            remove_index = self._ds.qcfilter.get_qc_test_mask(
                var_name=var_name,
                qc_var_name=qc_var_name,
                test_number=test_number,
                return_index=True,
            )
            self._ds.qcfilter.unset_test(
                var_name=var_name,
                qc_var_name=qc_var_name,
                index=remove_index,
                test_number=test_number,
                flag_value=flag_value,
            )
            if isinstance(flag_masks, list):
                del flag_masks[index]
            else:
                flag_masks = np.delete(flag_masks, index)
            self._ds[qc_var_name].attrs['flag_masks'] = flag_masks

        flag_meanings = self._ds[qc_var_name].attrs['flag_meanings']
        del flag_meanings[index]
        self._ds[qc_var_name].attrs['flag_meanings'] = flag_meanings

        flag_assessments = self._ds[qc_var_name].attrs['flag_assessments']
        del flag_assessments[index]
        self._ds[qc_var_name].attrs['flag_assessments'] = flag_assessments

    def set_test(self, var_name, index=None, test_number=None, flag_value=False):
        """
        Method to set a test/filter in a quality control variable.

        Parameters
        ----------
        var_name : str
            Data variable name.
        index : int or list or numpy array
            Index to set test in quality control array. If want to
            unset all values will need to pass in index of all values.
        test_number : int
            Test number to set.
        flag_value : boolean
            Switch to use flag_values integer quality control.

        Examples
        --------
            .. code-block:: python

                index = [0, 1, 2, 30]
                ds.qcfilter.set_test(var_name, index=index, test_number=2)

        """

        qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(var_name)

        qc_variable = np.array(self._ds[qc_var_name].values)

        # Ensure the qc_variable data type is integer. This ensures bitwise comparison
        # will not cause an error.
        if qc_variable.dtype.kind not in np.typecodes['AllInteger']:
            qc_variable = qc_variable.astype(int)

        # Determine if test number is too large for current data type. If so
        # up convert data type.
        dtype = qc_variable.dtype
        if np.iinfo(dtype).max - set_bit(0, test_number) < -1:
            if dtype == np.int8:
                dtype = np.int16
            elif dtype == np.int16:
                dtype = np.int32
            elif dtype == np.int32:
                dtype = np.int64

            qc_variable = qc_variable.astype(dtype)

        if index is not None:
            if flag_value:
                qc_variable[index] = test_number
            else:
                if bool(np.shape(index)):
                    qc_variable[index] = set_bit(qc_variable[index], test_number)
                elif index == 0:
                    qc_variable = set_bit(qc_variable, test_number)

        self._ds[qc_var_name].values = qc_variable

    def unset_test(
        self,
        var_name=None,
        qc_var_name=None,
        index=None,
        test_number=None,
        flag_value=False,
        flag_values_reset_value=0,
    ):
        """
        Method to unset a test/filter from a quality control variable.

        Parameters
        ----------
        var_name : str or None
            Data variable name.
        qc_var_name : str or None
            Quality control variable name. Ignored if var_name is set.
        index : int or list or numpy array
            Index to unset test in quality control array. If want to
            unset all values will need to pass in index of all values.
        test_number : int
            Test number to remove.
        flag_value : boolean
            Switch to use flag_values integer quality control.
        flag_values_reset_value : int
            Value to use when resetting a flag_values value to not be set.

        Examples
        --------
        .. code-block:: python

            ds.qcfilter.unset_test(var_name, index=range(10, 100), test_number=2)

        """
        if index is None:
            return

        if var_name is None and qc_var_name is None:
            raise ValueError(
                'You need to provide a value for var_name or qc_var_name '
                'keyword when calling the unset_test() method'
            )

        if var_name is not None:
            qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(var_name)

        # Get QC variable
        qc_variable = self._ds[qc_var_name].values

        # Ensure the qc_variable data type is integer. This ensures bitwise comparison
        # will not cause an error.
        if qc_variable.dtype.kind not in np.typecodes['AllInteger']:
            qc_variable = qc_variable.astype(int)

        if flag_value:
            qc_variable[index] = flag_values_reset_value
        else:
            qc_variable[index] = unset_bit(qc_variable[index], test_number)

        self._ds[qc_var_name].values = qc_variable

    def available_bit(self, qc_var_name, recycle=False):
        """
        Method to determine next available bit or flag to use with a QC test.
        Will check for flag_masks first and if not found will check for
        flag_values. This will drive how the next value is chosen.

        Parameters
        ----------
        qc_var_name : str
            Quality control variable name.
        recycle : boolean
            Option to look for a bit (test) not in use starting from 1.
            If a test is not defined will return the lowest number, else
            will just use next highest number.

        Returns
        -------
        test_num : int
            Next available test number.

        Examples
        --------
            .. code-block:: python

                from act.tests import EXAMPLE_METE40
                from act.io.arm import read_arm_netcdf
                ds = read_arm_netcdf(EXAMPLE_METE40, cleanup_qc=True)
                test_number = ds.qcfilter.available_bit('qc_atmos_pressure')
                print(test_number)


        """
        try:
            flag_masks = self._ds[qc_var_name].attrs['flag_masks']
            flag_value = False
        except KeyError:
            try:
                flag_masks = self._ds[qc_var_name].attrs['flag_values']
                flag_value = True
            except KeyError:
                try:
                    self._ds[qc_var_name].attrs['flag_values']
                    flag_masks = self._ds[qc_var_name].attrs['flag_masks']
                    flag_value = False
                except KeyError:
                    raise ValueError(
                        'Problem getting next value from '
                        'available_bit(). flag_values and '
                        'flag_masks not set as expected'
                    )

        if flag_masks == []:
            next_bit = 1
        else:
            if flag_value:
                if recycle:
                    next_bit = min(set(range(1, 100000)) - set(flag_masks))
                else:
                    next_bit = max(flag_masks) + 1
            else:
                if recycle:
                    tests = [parse_bit(mask)[0] for mask in flag_masks]
                    next_bit = min(set(range(1, 63)) - set(tests))
                else:
                    next_bit = parse_bit(max(flag_masks))[0] + 1

        return int(next_bit)

    def get_qc_test_mask(
        self,
        var_name=None,
        test_number=None,
        qc_var_name=None,
        flag_value=False,
        return_index=False,
    ):
        """
        Returns a numpy array of False or True where a particular
        flag or bit is set in a numpy array. Must set var_name or qc_var_name
        when calling.

        Parameters
        ----------
        var_name : str or None
            Data variable name.
        test_number : int
            Test number to return array where test is set.
        qc_var_name : str or None
            Quality control variable name. Ignored if var_name is set.
        flag_value : boolean
            Switch to use flag_values integer quality control.
        return_index : boolean
            Return a numpy array of index numbers into QC array where the
            test is set instead of False or True mask.

        Returns
        -------
        test_mask : numpy bool array or numpy integer array
            A numpy boolean array with False or True where the test number or
            bit was set, or numpy integer array of indexes where test is True.

        Examples
        --------
            .. code-block:: python

                from act.io.arm import read_arm_netcdf
                from act.tests import EXAMPLE_IRT25m20s

                ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
                var_name = "inst_up_long_dome_resist"
                result = ds.qcfilter.add_test(
                    var_name, index=[0, 1, 2], test_meaning="Birds!"
                )
                qc_var_name = result["qc_variable_name"]
                mask = ds.qcfilter.get_qc_test_mask(
                    var_name, result["test_number"], return_index=True
                )
                print(mask)
                array([0, 1, 2])

                mask = ds.qcfilter.get_qc_test_mask(var_name, result["test_number"])
                print(mask)
                array([True, True, True, ..., False, False, False])

                data = ds[var_name].values
                print(data[mask])
                array([7.84, 7.8777, 7.8965], dtype=float32)

                import numpy as np

                data[mask] = np.nan
                print(data)
                array([nan, nan, nan, ..., 7.6705, 7.6892, 7.6892], dtype=float32)

        """
        if var_name is None and qc_var_name is None:
            raise ValueError(
                'You need to provide a value for var_name or qc_var_name '
                'keyword when calling the get_qc_test_mask() method'
            )

        if test_number is None:
            raise ValueError(
                'You need to provide a value for test_number '
                'keyword when calling the get_qc_test_mask() method'
            )

        if var_name is not None:
            qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(var_name)

        qc_variable = self._ds[qc_var_name].values
        # Ensure the qc_variable data type is integer. This ensures bitwise comparison
        # will not cause an error.
        if qc_variable.dtype.kind not in np.typecodes['AllInteger']:
            qc_variable = qc_variable.astype(int)

        if flag_value:
            tripped = qc_variable == test_number
        else:
            check_bit = set_bit(0, test_number) & qc_variable
            tripped = check_bit > 0

        test_mask = np.full(qc_variable.shape, False, dtype='bool')
        # Make sure test_mask is an array. If qc_variable is scalar will
        # be retuned from np.zeros as scalar.
        test_mask = np.atleast_1d(test_mask)
        test_mask[tripped] = True
        test_mask = np.ma.make_mask(test_mask, shrink=False)

        if return_index:
            test_mask = np.where(test_mask)[0]

        return test_mask

    def get_masked_data(
        self,
        var_name,
        rm_assessments=None,
        rm_tests=None,
        return_nan_array=False,
        ma_fill_value=None,
        return_inverse=False,
    ):
        """
        Returns a numpy masked array containing data and mask or
        a numpy float array with masked values set to NaN.

        Parameters
        ----------
        var_name : str
            Data variable name.
        rm_assessments : str or list of str
            Assessment name to exclude from returned data.
        rm_tests : int or list of int
            Test numbers to exclude from returned data. This is the test
            number (or bit position number) not the mask number.
        return_nan_array : boolean
            Return a numpy array with filtered ( or masked) values
            set to numpy NaN value. If the data is type int will upconvert
            to numpy float to allow setting NaN value.
        ma_fill_value : int or float (or str?)
            The numpy masked array fill_value used in creation of the the
            masked array. If the datatype needs to be upconverted to allow
            the fill value to be used, data will be upconverted.
        return_inverse : boolean
            Invert the masked array mask or return data array where mask is set
            to False instead of True set to NaN. Useful for overplotting
            where failing.

        Returns
        -------
        variable : numpy masked array or numpy float array
            Default is to return a numpy masked array with the mask set to
            True where the test with requested assessment or test number
            was found set.
            If return_nan_array is True will return numpy array upconverted
            to float with locations where the test with requested assessment
            or test number was found set converted to NaN.

        Examples
        --------
            .. code-block:: python

                from act.io.arm import read_arm_netcdf
                from act.tests import EXAMPLE_IRT25m20s

                ds = read_arm_netcdf(EXAMPLE_IRT25m20s)
                var_name = "inst_up_long_dome_resist"
                result = ds.qcfilter.add_test(
                    var_name, index=[0, 1, 2], test_meaning="Birds!"
                )
                data = ds.qcfilter.get_masked_data(
                    var_name, rm_assessments=["Bad", "Indeterminate"]
                )
                print(data)
                masked_array(
                    data=[..., 7.670499801635742, 7.689199924468994, 7.689199924468994],
                    mask=[..., False, False, False],
                    fill_value=1e20,
                    dtype=float32,
                )

        """
        qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(var_name, add_if_missing=False)

        flag_value = False
        flag_values = None
        flag_masks = None
        flag_assessments = None
        try:
            flag_assessments = self._ds[qc_var_name].attrs['flag_assessments']
            flag_masks = self._ds[qc_var_name].attrs['flag_masks']
        except KeyError:
            pass

        try:
            flag_values = self._ds[qc_var_name].attrs['flag_values']
            flag_value = True
        except KeyError:
            pass

        test_numbers = []
        if rm_tests is not None:
            if isinstance(rm_tests, (int, float, str)):
                rm_tests = [int(rm_tests)]
            test_numbers.extend(rm_tests)

        if rm_assessments is not None:
            if isinstance(rm_assessments, str):
                rm_assessments = [rm_assessments]

            if flag_masks is not None:
                test_nums = [parse_bit(mask)[0] for mask in flag_masks]

            if flag_values is not None:
                test_nums = flag_values

            rm_assessments = [x.lower() for x in rm_assessments]
            if flag_assessments is not None:
                for ii, assessment in enumerate(flag_assessments):
                    if assessment.lower() in rm_assessments:
                        test_numbers.append(test_nums[ii])

        # Make the list of test numbers to mask unique
        test_numbers = list(set(test_numbers))

        # Create mask of indexes by looking where each test is set
        variable = self._ds[var_name].values
        nan_dtype = np.float32
        if variable.dtype in (np.float64, np.int64):
            nan_dtype = np.float64

        mask = np.zeros(variable.shape, dtype=bool)
        for test in test_numbers:
            mask = mask | self._ds.qcfilter.get_qc_test_mask(var_name, test, flag_value=flag_value)

        # Convert data numpy array into masked array
        try:
            variable = np.ma.array(variable, mask=mask, fill_value=ma_fill_value)
        except TypeError:
            variable = np.ma.array(
                variable,
                mask=mask,
                fill_value=ma_fill_value,
                dtype=np.array(ma_fill_value).dtype,
            )

        # If requested switch array from where data is not failing tests
        # to where data is failing tests. This can be used when over plotting
        # where the data if failing the tests.
        if return_inverse:
            mask = variable.mask
            mask = np.invert(mask)
            variable.mask = mask

        # If asked to return numpy array with values set to NaN
        if return_nan_array:
            variable = variable.astype(nan_dtype)
            variable = variable.filled(fill_value=np.nan)

        return variable

    def datafilter(
        self,
        variables=None,
        rm_assessments=None,
        rm_tests=None,
        verbose=False,
        del_qc_var=False,
    ):
        """
        Method to apply quality control variables to data variables by
        changing the data values in the dataset using quality control variables.
        The data is updated with failing data set to
        NaN. This can be used to update the data variable in the xarray
        dataset for use with xarray methods to perform analysis on the data
        since those methods don't read the quality control variables.

        Parameters
        ----------
        variables : None or str or list of str
            Data variable names to process. If set to None will update all
            data variables.
        rm_assessments : str or list of str
            Assessment names listed under quality control varible flag_assessments
            to exclude from returned data. Examples include
            ['Bad', 'Incorrect', 'Indeterminate', 'Suspect']
        rm_tests : int or list of int
            Test numbers listed under quality control variable to exclude from
            returned data. This is the test
            number (or bit position number) not the mask number.
        verbose : boolean
            Print processing information.
        del_qc_var : boolean
            Option to delete quality control variable after processing. Since
            the data values can not be determined after they are set to NaN
            and xarray method processing would also process the quality control
            variables, the default is to remove the quality control data
            variables.  Defaults to False.

        Examples
        --------
            .. code-block:: python

                from act.io.arm import read_arm_netcdf
                from act.tests import EXAMPLE_MET1

                ds = read_arm_netcdf(EXAMPLE_MET1)
                ds.clean.cleanup()

                var_name = "atmos_pressure"

                ds_1 = ds.nanmean()

                ds.qcfilter.add_less_test(var_name, 99, test_assessment="Bad")
                ds.qcfilter.datafilter(rm_assessments="Bad")
                ds_2 = ds.nanmean()

                print("All_data =", ds_1[var_name].values)
                All_data = 98.86098
                print("Bad_Removed =", ds_2[var_name].values)
                Bad_Removed = 99.15148

        """

        if rm_assessments is None and rm_tests is None:
            raise ValueError('Need to set rm_assessments or rm_tests option')

        if variables is not None and isinstance(variables, str):
            variables = [variables]

        if variables is None:
            variables = list(self._ds.data_vars)

        for var_name in variables:
            qc_var_name = self.check_for_ancillary_qc(var_name, add_if_missing=False, cleanup=False)
            if qc_var_name is None:
                if verbose:
                    if var_name in ['base_time', 'time_offset']:
                        continue

                    try:
                        if self._ds[var_name].attrs['standard_name'] == 'quality_flag':
                            continue
                    except KeyError:
                        pass

                    print(
                        f'No quality control variable for {var_name} found '
                        f'in call to .qcfilter.datafilter()'
                    )

                continue

            # Need to return data as Numpy array with NaN values. Setting the Dask array
            # to Numpy masked array does not work with other tools.
            data = self.get_masked_data(
                var_name, rm_assessments=rm_assessments, rm_tests=rm_tests, return_nan_array=True
            )

            # If data was orginally stored as Dask array return values to Dataset as Dask array
            # else set as Numpy array.
            try:
                self._ds[var_name].data = dask.array.from_array(
                    data, chunks=self._ds[var_name].data.chunksize
                )

            except AttributeError:
                self._ds[var_name].values = data

            # Adding information on filtering to history attribute
            flag_masks = None
            flag_assessments = None
            flag_meanings = None
            try:
                flag_assessments = list(self._ds[qc_var_name].attrs['flag_assessments'])
                flag_masks = list(self._ds[qc_var_name].attrs['flag_masks'])
                flag_meanings = list(self._ds[qc_var_name].attrs['flag_meanings'])
            except KeyError:
                pass

            # Add comment to history for each test that's filtered out
            if isinstance(rm_tests, int):
                rm_tests = [rm_tests]
            if rm_tests is not None:
                for test in list(rm_tests):
                    test = 2 ** (test - 1)
                    if test in flag_masks:
                        index = flag_masks.index(test)
                        comment = ''.join(['act.qc.datafilter: ', flag_meanings[index]])
                        if 'history' in self._ds[var_name].attrs.keys():
                            self._ds[var_name].attrs['history'] += '\n' + comment
                        else:
                            self._ds[var_name].attrs['history'] = comment

            if isinstance(rm_assessments, str):
                rm_assessments = [rm_assessments]
            if rm_assessments is not None:
                for assessment in rm_assessments:
                    if assessment in flag_assessments:
                        index = [i for i, e in enumerate(flag_assessments) if e == assessment]
                        for ind in index:
                            comment = ''.join(['act.qc.datafilter: ', flag_meanings[ind]])
                            if 'history' in self._ds[var_name].attrs.keys():
                                self._ds[var_name].attrs['history'] += '\n' + comment
                            else:
                                self._ds[var_name].attrs['history'] = comment

            # If requested delete quality control variable
            if del_qc_var:
                del self._ds[qc_var_name]
                if verbose:
                    print(f'Deleting {qc_var_name} from dataset')


def set_bit(array, bit_number):
    """
    Function to set a quality control bit given a scalar or
    array of values and a bit number.

    Parameters
    ----------
    array : int list of int or numpy array of int
        The bitpacked array to set the bit number.
    bit_number : int
        The bit (or test) number to set starting at 1.

    Returns
    -------
    array : int, numpy array, tuple, list
        Integer or numpy array with bit set for each element of the array.
        Returned in same type.

    Examples
    --------
    Example use setting bit 2 to an array called data:

        .. code-block:: python

            from act.qc.qcfilter import set_bit
            data = np.array(range(0, 7))
            data = set_bit(data, 2)
            print(data)
            array([2, 3, 2, 3, 6, 7, 6])

    """
    was_list = False
    was_tuple = False
    if isinstance(array, list):
        array = np.array(array)
        was_list = True

    if isinstance(array, tuple):
        array = np.array(array)
        was_tuple = True

    if bit_number > 0:
        array |= 1 << bit_number - 1

    if was_list:
        array = list(array)

    if was_tuple:
        array = tuple(array)

    return array


def unset_bit(array, bit_number):
    """
    Function to remove a quality control bit given a
    scalar or array of values and a bit number.

    Parameters
    ----------
    array : int list of int or numpy array
        Array of integers containing bit packed numbers.
    bit_number : int
        Bit number to remove starting at 1.

    Returns
    -------
    array : int or numpy array
        Returns same data type as array entered with bit removed. Will
        fail gracefully if the bit requested to be removed was not set.

    Examples
    --------
       .. code-block:: python

            from act.qc.qcfilter import set_bit, unset_bit
            data = set_bit([0, 1, 2, 3, 4], 2)
            data = set_bit(data, 3)
            print(data)
            [6, 7, 6, 7, 6]

            data = unset_bit(data, 2)
            print(data)
            [4, 5, 4, 5, 4]

    """
    was_list = False
    was_tuple = False
    if isinstance(array, list):
        array = np.array(array)
        was_list = True

    if isinstance(array, tuple):
        array = np.array(array)
        was_tuple = True

    if bit_number > 0:
        array &= ~(1 << bit_number - 1)

    if was_list:
        array = list(array)

    if was_tuple:
        array = tuple(array)

    return array


def parse_bit(qc_bit):
    """
    Given a single integer value, return bit positions.

    Parameters
    ----------
    qc_bit : int or numpy int
        Bit packed integer number to be parsed.

    Returns
    -------
    bit_number : numpy.int32 array
        Array containing all bit numbers of the bit packed number.
        If no bits set returns empty array.

    Examples
    --------
        .. code-block:: python

            from act.qc.qcfilter import parse_bit
            parse_bit(7)
            array([1, 2, 3], dtype=int32)

    """
    if isinstance(qc_bit, (list, tuple, np.ndarray)):
        if len(qc_bit) > 1:
            raise ValueError('Must be a single value.')
        qc_bit = qc_bit[0]

    if qc_bit < 0:
        raise ValueError('Must be a positive integer.')

    # Convert integer value to single element numpy array of type unsigned integer 64
    value = np.array([qc_bit]).astype(">u8")

    # Convert value to view containing only unsigned integer 8 data type. This
    # is required for the numpy unpackbits function which only works with
    # unsigned integer 8 bit data type.
    value = value.view("u1")

    # Unpack bits using numpy into array of 1 where bit is set and convert into boolean array
    index = np.unpackbits(value).astype(bool)

    # Create range of numbers from 64 to 1 and subset where unpackbits found a bit set.
    bit_number = np.arange(index.size, 0, -1)[index]

    # Flip the array to increasing numbers to match historical method
    bit_number = np.flip(bit_number)

    # bit_number = []
    # qc_bit = int(qc_bit)

    # counter = 0
    # while qc_bit > 0:
    #     temp_value = qc_bit % 2
    #     qc_bit = qc_bit >> 1
    #     counter += 1
    #     if temp_value == 1:
    #         bit_number.append(counter)

    # Convert data type into expected type
    bit_number = np.asarray(bit_number, dtype=np.int32)

    return bit_number
