"""
Class definitions for cleaning up QC variables to standard
cf-compliance.

"""

import copy
import re

import numpy as np
import xarray as xr

from act.qc.qcfilter import parse_bit


@xr.register_dataset_accessor('clean')
class CleanDataset:
    """
    Class for cleaning up QC variables to standard cf-compliance
    """

    def __init__(self, ds):
        self._ds = ds

    @property
    def matched_qc_variables(self, check_arm_syntax=True):
        """
        Find variables that are QC variables and return list of names.

        Parameters
        ----------
        check_arm_syntax : boolean
            ARM ueses a standard of starting all quality control variables
            with "qc" joined with an underscore. This is a more robust method
            of getting the quality control variables before the standard_name
            attribute is added. If this is true will first check using
            attributes and will then check if variable starts with "qc".

        Returns
        -------
        variables : list of str
            A list of strings containing the name of each variable.

        """

        # Will need to find all historical cases and add to list
        description_list = [
            'See global attributes for individual.+bit descriptions.',
            (
                'This field contains bit packed integer values, where each '
                'bit represents a QC test on the data. Non-zero bits indicate '
                'the QC condition given in the description for those bits; '
                'a value of 0.+ indicates the data has not '
                'failed any QC tests.'
            ),
            (r'This field contains bit packed values which should be ' r'interpreted as listed..+'),
        ]

        # Loop over each variable and look for a match to an attribute that
        # would exist if the variable is a QC variable.
        variables = []
        for var in self._ds.data_vars:
            try:
                if self._ds[var].attrs['standard_name'] == 'quality_flag':
                    variables.append(var)
                    continue
            except KeyError:
                pass

            if check_arm_syntax and var.startswith('qc_'):
                variables.append(var)
                continue

            try:
                for desc in description_list:
                    if re.match(desc, self._ds[var].attrs['description']) is not None:
                        variables.append(var)
                        break
            except KeyError:
                pass

        variables = list(set(variables))

        return variables

    def cleanup(
        self,
        cleanup_arm_qc=True,
        clean_arm_state_vars=None,
        handle_missing_value=True,
        link_qc_variables=True,
        normalize_assessment=False,
        cleanup_cf_qc=True,
        **kwargs,
    ):
        """
        Wrapper method to automatically call all the standard methods
        for dataset cleanup.

        Parameters
        ----------
        cleanup_arm_qc : bool
            Option to clean Xarray dataset from ARM QC to CF QC standards.
        clean_arm_state_vars : list of str
            Option to clean Xarray dataset state variables from ARM to CF
            standards. Pass in list of variable names.
        handle_missing_value : bool
            Go through variables and look for cases where a QC or state varible
            was convereted to a float and missing values set to np.nan. This
            is done because of xarry's default to use mask_and_scale=True.
            This will convert the data type back to integer and replace
            any instances of np.nan to a missing value indicator (most
            likely -9999).
        link_qc_variables : bool
            Option to link QC variablers through ancillary_variables if not
            already set.
        normalize_assessment : bool
            Option to clean up assessments to use the same terminology. Set to
            False for default because should only be an issue after adding DQRs
            and the function to add DQRs calls this method.
        **kwargs : keywords
            Keyword arguments passed through to clean.clean_arm_qc
            method.

        Examples
        --------
            .. code-block:: python

                files = act.tests.sample_files.EXAMPLE_MET1
                ds = act.io.arm.read_arm_netcdf(files)
                ds.clean.cleanup()

        """
        # Convert ARM QC to be more like CF state fields
        if cleanup_arm_qc:
            self._ds.clean.clean_arm_qc(**kwargs)

        # Convert ARM state fields to be more liek CF state fields
        if clean_arm_state_vars is not None:
            self._ds.clean.clean_arm_state_variables(clean_arm_state_vars)

        # Correctly convert data type because of missing value
        # indicators in state and QC variables. Needs to be run after
        # clean.clean_arm_qc to use CF attribute names.
        if handle_missing_value:
            self._ds.clean.handle_missing_values()

        # Add some ancillary_variables linkages
        # between data variable and QC variable
        if link_qc_variables:
            self._ds.clean.link_variables()

        # Update the terminology used with flag_assessments to be consistent
        if normalize_assessment:
            self._ds.clean.normalize_assessment()

        # Update from CF to standard used in ACT
        if cleanup_cf_qc:
            self._ds.clean.clean_cf_qc(**kwargs)

    def handle_missing_values(self, default_missing_value=np.int32(-9999)):
        """
        Correctly handle missing_value and _FillValue in the dataset.
        xarray will automatically replace missing_value and
        _FillValue in the data with NaN. This is great for data set
        as type float but not great for int data. Can cause issues
        with QC and state fields. This will loop through the array
        looking for state and QC fields and revert them back to int
        data type if upconverted to float to handle NaNs. Issue is that
        xarray will convert data type to float if the attribute is defined
        even if no data are set as missing value. xarray will also then
        remove the missing_value or _FillValue variable attribute. This
        will put the missing_value attribute back if needed.

        Parameters
        ----------
        default_missing_value : numpy int or float
           The default missing value to use if a missing_value attribute
           is not defined but one is needed.

        """
        state_att_names = [
            'flag_values',
            'flag_meanings',
            'flag_masks',
            'flag_attributes',
        ]

        # Look for variables that have 2 of the state_att_names defined
        # as attribures and is of type float. If so assume the variable
        # was incorreclty converted to float type.
        for var in self._ds.data_vars:
            var_att_names = self._ds[var].attrs.keys()
            if len(set(state_att_names) & set(var_att_names)) >= 2 and self._ds[
                var
            ].values.dtype in [
                np.dtype('float16'),
                np.dtype('float32'),
                np.dtype('float64'),
            ]:
                # Look at units variable to see if this is the stupid way some
                # ARM products mix data and state variables. If the units are not
                # in the normal list of unitless type assume this is a data variable
                # and skip. Other option is to lookf or a valid_range attribute
                # and skip. This is commented out for now since the units check
                # appears to be working.
                try:
                    if self._ds[var].attrs['units'] not in ['1', 'unitless', '', ' ']:
                        continue
                except KeyError:
                    pass

                # Change any np.nan values to missing value indicator
                data = self._ds[var].values
                data[np.isnan(data)] = default_missing_value.astype(data.dtype)

                # Convert data to match type of flag_mask or flag_values
                # as the best guess of what type is correct.
                found_dtype = False
                for att_name in ['flag_masks', 'flag_values']:
                    try:
                        att_value = self._ds[var].attrs[att_name]
                        if isinstance(att_value, (list, tuple)):
                            dtype = att_value[0].dtype
                        elif isinstance(att_value, str):
                            dtype = default_missing_value.dtype
                            att_value = att_value.replace(',', ' ').split()
                            att_value = np.array(att_value, dtype=dtype)
                            self._ds[var].attrs[att_name] = att_value
                            dtype = default_missing_value.dtype
                        else:
                            dtype = att_value.dtype
                        data = data.astype(dtype)
                        found_dtype = True
                        break
                    except (KeyError, IndexError, AttributeError):
                        pass

                # If flag_mask or flag_values is not available choose an int type
                # and set data to that type.
                if found_dtype is False:
                    data = data.astype(default_missing_value.dtype)

                # Return data to the dataset and add missing value indicator
                # attribute to variable.
                self._ds[var].values = data
                self._ds[var].attrs['missing_value'] = default_missing_value.astype(data.dtype)

    def get_attr_info(self, variable=None, flag=False):
        """
        Get ARM quality control definitions from the ARM standard
        bit_#_description, ... attributes and return as dictionary.
        Will attempt to guess if the flag is integer or bit packed
        based on what attributes are set.

        Parameters
        ----------
        variable : str
            Variable name to get attribute information. If set to None
            will get global attributes.
        flag : bool
            Optional flag indicating if QC is expected to be bitpacked
            or integer. Flag = True indicates integer QC. Default
            is bitpacked or False.

        Returns
        -------
        attributes dictionary : dict or None
            A dictionary contianing the attribute information converted from
            ARM QC to CF QC. All keys include 'flag_meanings', 'flag_masks',
            'flag_values', 'flag_assessments', 'flag_tests', 'arm_attributes'.
            Returns None if none found.

        """
        string = 'bit'
        if flag:
            string = 'flag'
        else:
            found_string = False
            try:
                if self._ds.attrs['qc_bit_comment']:
                    string = 'bit'
                    found_string = True
            except KeyError:
                pass

            if found_string is False:
                try:
                    if self._ds.attrs['qc_flag_comment']:
                        string = 'flag'
                        found_string = True
                except KeyError:
                    pass

            if found_string is False:
                var = self.matched_qc_variables
                if len(var) > 0:
                    try:
                        if self._ds[variable].attrs['flag_method'] == 'integer':
                            string = 'flag'
                        found_string = True
                        del self._ds[variable].attrs['flag_method']
                    except KeyError:
                        pass

        try:
            if variable:
                attr_description_pattern = r'(^' + string + r')_([0-9]+)_(description$)'
                attr_assessment_pattern = r'(^' + string + r')_([0-9]+)_(assessment$)'
                attr_comment_pattern = r'(^' + string + r')_([0-9]+)_(comment$)'
                attributes = self._ds[variable].attrs
            else:
                attr_description_pattern = r'(^qc_' + string + r')_([0-9]+)_(description$)'
                attr_assessment_pattern = r'(^qc_' + string + r')_([0-9]+)_(assessment$)'
                attr_comment_pattern = r'(^qc_' + string + r')_([0-9]+)_(comment$)'
                attributes = self._ds.attrs
        except KeyError:
            return None

        assessment_bit_num = []
        description_bit_num = []
        comment_bit_num = []
        flag_masks = []
        flag_meanings = []
        flag_assessments = []
        flag_comments = []
        arm_attributes = []

        dtype = np.int32
        for att_name in attributes:
            try:
                description = re.match(attr_description_pattern, att_name)
                description_bit_num.append(int(description.groups()[1]))
                flag_meanings.append(attributes[att_name])
                arm_attributes.append(att_name)
            except AttributeError:
                pass

            try:
                assessment = re.match(attr_assessment_pattern, att_name)
                assessment_bit_num.append(int(assessment.groups()[1]))
                flag_assessments.append(attributes[att_name])
                arm_attributes.append(att_name)
            except AttributeError:
                pass

            try:
                comment = re.match(attr_comment_pattern, att_name)
                comment_bit_num.append(int(comment.groups()[1]))
                flag_comments.append(attributes[att_name])
                arm_attributes.append(att_name)
            except AttributeError:
                pass

        if variable is not None:
            # Try and get the data type from the variable if it is an integer
            # If not an integer make the flag values integers.
            try:
                dtype = self._ds[variable].values.dtype
                if np.issubdtype(dtype, np.integer):
                    pass
                else:
                    dtype = np.int32
            except AttributeError:
                pass

        # Sort on bit number to ensure correct description order
        index = np.argsort(description_bit_num)
        flag_meanings = np.array(flag_meanings)
        description_bit_num = np.array(description_bit_num)
        flag_meanings = flag_meanings[index]
        description_bit_num = description_bit_num[index]

        # Sort on bit number to ensure correct assessment order
        if len(flag_assessments) > 0:
            if len(flag_assessments) < len(flag_meanings):
                for ii in range(1, len(flag_meanings) + 1):
                    if ii not in assessment_bit_num:
                        assessment_bit_num.append(ii)
                        flag_assessments.append('')
            index = np.argsort(assessment_bit_num)
            flag_assessments = np.array(flag_assessments)
            flag_assessments = flag_assessments[index]

        # Sort on bit number to ensure correct comment order
        if len(flag_comments) > 0:
            if len(flag_comments) < len(flag_meanings):
                for ii in range(1, len(flag_meanings) + 1):
                    if ii not in comment_bit_num:
                        comment_bit_num.append(ii)
                        flag_comments.append('')
            index = np.argsort(comment_bit_num)
            flag_comments = np.array(flag_comments)
            flag_comments = flag_comments[index]

        # Convert bit number to mask number
        if len(description_bit_num) > 0:
            flag_masks = np.array(description_bit_num)
            flag_masks = np.left_shift(1, flag_masks - 1)

        # build dictionary to return values
        if len(flag_masks) > 0 or len(description_bit_num) > 0:
            return_dict = dict()
            return_dict['flag_meanings'] = list(np.array(flag_meanings, dtype=str))

            if len(flag_masks) > 0 and max(flag_masks) > np.iinfo(np.uint32).max:
                flag_mask_dtype = np.uint64
            else:
                flag_mask_dtype = np.uint32

            if flag:
                return_dict['flag_values'] = list(np.array(description_bit_num, dtype=dtype))
                return_dict['flag_masks'] = list(np.array([], dtype=flag_mask_dtype))
            else:
                return_dict['flag_values'] = list(np.array([], dtype=dtype))
                return_dict['flag_masks'] = list(np.array(flag_masks, dtype=flag_mask_dtype))

            return_dict['flag_assessments'] = list(np.array(flag_assessments, dtype=str))
            return_dict['flag_tests'] = list(np.array(description_bit_num, dtype=dtype))
            return_dict['flag_comments'] = list(np.array(flag_comments, dtype=str))
            return_dict['arm_attributes'] = arm_attributes

        else:
            # If nothing to return set to None
            return_dict = None

        # If no QC is found but there's a Mentor_QC_Field_Information global attribute,
        # hard code the information.  This is for older ARM files that had QC information
        # in this global attribute.  For these cases, this should hold 100%
        if return_dict is None and 'Mentor_QC_Field_Information' in self._ds.attrs:
            qc_att = self._ds.attrs['Mentor_QC_Field_Information']
            if 'Basic mentor QC checks' in qc_att:
                if len(qc_att) == 920 or len(qc_att) == 1562:
                    return_dict = dict()
                    return_dict['flag_meanings'] = [
                        'Value is equal to missing_value.',
                        'Value is less than the valid_min.',
                        'Value is greater than the valid_max.',
                        'Difference between current and previous values exceeds valid_delta.',
                    ]
                    return_dict['flag_tests'] = [1, 2, 3, 4]
                    return_dict['flag_masks'] = [1, 2, 4, 8]
                    return_dict['flag_assessments'] = ['Bad', 'Bad', 'Bad', 'Indeterminate']
                    return_dict['flag_values'] = []
                    return_dict['flag_comments'] = []
                    return_dict['arm_attributes'] = [
                        'bit_1_description',
                        'bit_1_assessment',
                        'bit_2_description',
                        'bit_2_assessment',
                        'bit_3_description',
                        'bit_3_assessment',
                        'bit_4_description',
                        'bit_4_assessment',
                    ]

        return return_dict

    def clean_arm_state_variables(
        self,
        variables,
        override_cf_flag=True,
        clean_units_string=True,
        integer_flag=True,
        replace_in_flag_meanings=None,
    ):
        """
        Function to clean up state variables to use more CF style.

        Parameters
        ----------
        variables : str or list of str
            List of variable names to update.
        override_cf_flag : bool
            Option to overwrite CF flag_meanings attribute if it exists
            with the values from ARM QC bit_#_description.
        clean_units_string : bool
            Option to update units string if set to 'unitless' to be
            udunits compliant '1'.
        integer_flag : bool
            Pass through keyword of 'flag' for get_attr_info().
        replace_in_flag_meanings : None or string
            Character string to search and replace in each flag meanings array value
            to increase readability since the flag_meanings stored in netCDF file
            is a single character array separated by space character. Alows for
            replacing things like "_" with space character.

        """
        if isinstance(variables, str):
            variables = [variables]

        for var in variables:
            flag_info = self.get_attr_info(variable=var, flag=integer_flag)
            if flag_info is not None:
                # Add new attributes to variable
                for attr in ['flag_values', 'flag_meanings', 'flag_masks']:
                    if len(flag_info[attr]) > 0:
                        # Only add if attribute does not exist.
                        if attr in self._ds[var].attrs.keys() is False:
                            self._ds[var].attrs[attr] = copy.copy(flag_info[attr])
                        # If flag is set, set attribure even if exists
                        elif override_cf_flag:
                            self._ds[var].attrs[attr] = copy.copy(flag_info[attr])

                # Remove replaced attributes
                arm_attributes = flag_info['arm_attributes']
                for attr in arm_attributes:
                    try:
                        del self._ds[var].attrs[attr]
                    except KeyError:
                        pass

            # Check if flag_meanings is string. If so convert to list.
            try:
                flag_meanings = copy.copy(self._ds[var].attrs['flag_meanings'])
                if isinstance(flag_meanings, str):
                    flag_meanings = flag_meanings.split()
                    if replace_in_flag_meanings is not None:
                        for ii, flag_meaning in enumerate(flag_meanings):
                            flag_meaning = flag_meaning.replace(replace_in_flag_meanings, ' ')
                            flag_meanings[ii] = flag_meaning

                    self._ds[var].attrs['flag_meanings'] = flag_meanings
            except KeyError:
                pass

            # Clean up units attribute from unitless to udunits '1'
            try:
                if clean_units_string and self._ds[var].attrs['units'] == 'unitless':
                    self._ds[var].attrs['units'] = '1'
            except KeyError:
                pass

    def correct_valid_minmax(self, qc_variable):
        """
        Function to correct the name and location of quality control limit
        variables that use valid_min and valid_max incorrectly.

        Parameters
        ----------
        qc_variable : str
            Name of quality control variable in the Xarray dataset to correct.

        """
        test_dict = {
            'valid_min': 'fail_min',
            'valid_max': 'fail_max',
            'valid_delta': 'fail_delta',
        }

        aa = re.match(r'^qc_(.+)', qc_variable)
        variable = None
        try:
            variable = aa.groups()[0]
        except AttributeError:
            return

        made_change = False
        try:
            flag_meanings = copy.copy(self._ds[qc_variable].attrs['flag_meanings'])
        except KeyError:
            return

        for attr in test_dict.keys():
            for ii, test in enumerate(flag_meanings):
                if attr in test:
                    flag_meanings[ii] = re.sub(attr, test_dict[attr], test)
                    made_change = True
                    try:
                        self._ds[qc_variable].attrs[test_dict[attr]] = copy.copy(
                            self._ds[variable].attrs[attr]
                        )
                        del self._ds[variable].attrs[attr]
                    except KeyError:
                        pass

        if made_change:
            self._ds[qc_variable].attrs['flag_meanings'] = flag_meanings

    def link_variables(self):
        """
        Add some attributes to link and explain data
        to QC data relationship. Will use non-CF standard_name
        of quality_flag. Hopefully this will be added to the
        standard_name table in the future.
        """
        for var in self._ds.data_vars:
            aa = re.match(r'^qc_(.+)', var)
            try:
                variable = aa.groups()[0]
                qc_variable = var
            except AttributeError:
                continue
            # Skip data quality fields.
            try:
                if "Quality check results on field:" not in self._ds[var].attrs["long_name"]:
                    continue
            except KeyError:
                pass

            # Get existing data variable ancillary_variables attribute
            try:
                ancillary_variables = self._ds[variable].attrs['ancillary_variables']
            except KeyError:
                ancillary_variables = ''

            # If the QC variable is not in ancillary_variables add
            if qc_variable not in ancillary_variables:
                ancillary_variables = qc_variable
            self._ds[variable].attrs['ancillary_variables'] = copy.copy(ancillary_variables)

            # Check if QC variable has correct standard_name and iff not fix it.
            correct_standard_name = 'quality_flag'
            try:
                if self._ds[qc_variable].attrs['standard_name'] != correct_standard_name:
                    self._ds[qc_variable].attrs['standard_name'] = correct_standard_name
            except KeyError:
                self._ds[qc_variable].attrs['standard_name'] = correct_standard_name

    def clean_arm_qc(
        self,
        override_cf_flag=True,
        clean_units_string=True,
        correct_valid_min_max=True,
        remove_unset_global_tests=True,
        **kwargs,
    ):
        """
        Method to clean up Xarray dataset QC variables.

        Parameters
        ----------
        override_cf_flag : bool
            Option to overwrite CF flag_masks, flag_meanings, flag_values
            if exists.
        clean_units_string : bool
            Option to clean up units string from 'unitless'
            to udunits compliant '1'.
        correct_valid_min_max : bool
            Option to correct use of valid_min and valid_max with QC variables
            by moving from data variable to QC varible, renaming to fail_min,
            fail_max and fail_detla if the valid_min, valid_max or valid_delta
            is listed in bit discription attribute. If not listed as
            used with QC will assume is being used correctly.
        remove_unset_global_tests : bool
            Option to look for globaly defined tests that are not set at the
            variable level and remove from quality control variable.

        """
        global_qc = self.get_attr_info()
        for qc_var in self.matched_qc_variables:
            # Clean up units attribute from unitless to udunits '1'
            try:
                if clean_units_string and self._ds[qc_var].attrs['units'] == 'unitless':
                    self._ds[qc_var].attrs['units'] = '1'
            except KeyError:
                pass

            qc_attributes = self.get_attr_info(variable=qc_var)

            if qc_attributes is None:
                qc_attributes = global_qc

            # Add new attributes to variable
            for attr in [
                'flag_masks',
                'flag_meanings',
                'flag_assessments',
                'flag_values',
                'flag_comments',
            ]:
                if qc_attributes is not None and len(qc_attributes[attr]) > 0:
                    # Only add if attribute does not exists
                    if attr in self._ds[qc_var].attrs.keys() is False:
                        self._ds[qc_var].attrs[attr] = copy.copy(qc_attributes[attr])
                    # If flag is set add attribure even if already exists
                    elif override_cf_flag:
                        self._ds[qc_var].attrs[attr] = copy.copy(qc_attributes[attr])

            # Remove replaced attributes
            if qc_attributes is not None:
                arm_attributes = qc_attributes['arm_attributes']
                if 'description' not in arm_attributes:
                    arm_attributes.append('description')
                if 'flag_method' not in arm_attributes:
                    arm_attributes.append('flag_method')
                for attr in arm_attributes:
                    try:
                        del self._ds[qc_var].attrs[attr]
                    except KeyError:
                        pass

            # Check for use of valid_min and valid_max as QC limits and fix
            if correct_valid_min_max:
                self._ds.clean.correct_valid_minmax(qc_var)

        # Clean up global attributes
        if global_qc is not None:
            global_attributes = global_qc['arm_attributes']
            global_attributes.extend(['qc_bit_comment'])
            for attr in global_attributes:
                try:
                    del self._ds.attrs[attr]
                except KeyError:
                    pass

        # If requested remove tests at variable level that were set from global level descriptions.
        # This is assuming the test was only performed if the limit value is listed with the variable
        # even if the global level describes the test.
        if remove_unset_global_tests and global_qc is not None:
            limit_name_list = ['fail_min', 'fail_max', 'fail_delta']

            for qc_var_name in self.matched_qc_variables:
                flag_meanings = self._ds[qc_var_name].attrs['flag_meanings']
                flag_masks = self._ds[qc_var_name].attrs['flag_masks']
                tests_to_remove = []
                for ii, flag_meaning in enumerate(flag_meanings):
                    # Loop over usual test attribute names looking to see if they
                    # are listed in test description. If so use that name for look up.
                    test_attribute_limit_name = None
                    for name in limit_name_list:
                        if name in flag_meaning:
                            test_attribute_limit_name = name
                            break

                    if test_attribute_limit_name is None:
                        continue

                    remove_test = True
                    test_number = parse_bit(flag_masks[ii])[0]
                    for attr_name in self._ds[qc_var_name].attrs:
                        if test_attribute_limit_name == attr_name:
                            remove_test = False
                            break

                        index = self._ds.qcfilter.get_qc_test_mask(
                            qc_var_name=qc_var_name, test_number=test_number
                        )
                        if np.any(index):
                            remove_test = False
                            break

                    if remove_test:
                        tests_to_remove.append(test_number)

                if len(tests_to_remove) > 0:
                    for test_to_remove in tests_to_remove:
                        self._ds.qcfilter.remove_test(
                            qc_var_name=qc_var_name, test_number=test_to_remove
                        )

    def normalize_assessment(
        self,
        variables=None,
        exclude_variables=None,
        qc_lookup={'Incorrect': 'Bad', 'Suspect': 'Indeterminate'},
    ):
        """
        Method to clean up assessment terms used to be consistent between
        embedded QC and DQRs.

        Parameters
        ----------
        variables : str or list of str
            Optional data variable names to check and normalize. If set to
            None will check all variables.
        exclude_variables : str or list of str
            Optional data variable names to exclude from processing.
        qc_lookup : dict
            Optional dictionary used to convert between terms.

        Examples
        --------
            .. code-block:: python

                ds = act.io.arm.read_arm_netcdf(files)
                ds.clean.normalize_assessment(variables='temp_mean')

            .. code-block:: python

                ds = act.io.arm.read_arm_netcdf(files, cleanup_qc=True)
                ds.clean.normalize_assessment(qc_lookup={'Bad': 'Incorrect', 'Indeterminate': 'Suspect'})

        """

        # Get list of variables if not provided
        if variables is None:
            variables = list(self._ds.data_vars)

        # Ensure variables is a list
        if not isinstance(variables, (list, tuple)):
            variables = [variables]

        # If exclude variables provided remove from variables list
        if exclude_variables is not None:
            if not isinstance(exclude_variables, (list, tuple)):
                exclude_variables = [exclude_variables]

            variables = list(set(variables) - set(exclude_variables))

        # Loop over variables checking if a QC variable exits and use the
        # lookup dictionary to convert the assessment terms.
        for var_name in variables:
            qc_var_name = self._ds.qcfilter.check_for_ancillary_qc(
                var_name, add_if_missing=False, cleanup=False
            )
            if qc_var_name is not None:
                try:
                    flag_assessments = self._ds[qc_var_name].attrs['flag_assessments']
                except KeyError:
                    continue

                for ii, assess in enumerate(flag_assessments):
                    try:
                        flag_assessments[ii] = qc_lookup[assess]
                    except KeyError:
                        continue

    def clean_cf_qc(self, variables=None, sep='__', **kwargs):
        """
        Method to convert the CF standard for QC attributes to match internal
        format expected in the Dataset. CF does not allow string attribute
        arrays, even though netCDF4 does allow string attribute arrays. The quality
        control variables uses and expects lists for flag_meaning, flag_assessments.

        Parameters
        ----------
        variables : str or list of str or None
            Data variable names to convert. If set to None will check all variables.
        sep : str or None
            Separater to use for splitting individual test meanings. Since the CF
            attribute in the netCDF file must be a string and is separated by a
            space character, individual test meanings are connected with a character.
            Default for ACT writing to file is double underscore to preserve underscores
            in variable or attribute names.
        kwargs : dict
            Additional keyword argumnts not used. This is to allow calling multiple
            methods from one method without causing unexpected keyword errors.

        Examples
        --------
            .. code-block:: python

                ds = act.io.arm.read_arm_netcdf(files)
                ds.clean.clean_cf_qc(variables='temp_mean')

            .. code-block:: python

                ds = act.io.arm.read_arm_netcdf(files, cleanup_qc=True)

        """

        # Convert string in to list of string for itteration
        if isinstance(variables, str):
            variables = [variables]

        # If no variables provided, get list of all variables in Dataset
        if variables is None:
            variables = list(self._ds.data_vars)

        for var_name in variables:
            # Check flag_meanings type. If string separate on space character
            # into list. If sep is not None split string on separater to make
            # better looking list of strings.
            try:
                flag_meanings = self._ds[var_name].attrs['flag_meanings']
                if isinstance(flag_meanings, str):
                    flag_meanings = flag_meanings.split()
                    if sep is not None:
                        flag_meanings = [ii.replace(sep, ' ') for ii in flag_meanings]
                    self._ds[var_name].attrs['flag_meanings'] = flag_meanings
            except KeyError:
                pass

            # Check if flag_assessments is a string, split on space character
            # to make list.
            try:
                flag_assessments = self._ds[var_name].attrs['flag_assessments']
                if isinstance(flag_assessments, str):
                    flag_assessments = flag_assessments.split()
                    self._ds[var_name].attrs['flag_assessments'] = flag_assessments
            except KeyError:
                pass

            # Check if flag_masks is a numpy scalar instead of array. If so convert
            # to numpy array. If value is not numpy scalar, turn single value into
            # list.
            try:
                flag_masks = self._ds[var_name].attrs['flag_masks']
                if type(flag_masks).__module__ == 'numpy':
                    if flag_masks.shape == ():
                        self._ds[var_name].attrs['flag_masks'] = np.atleast_1d(flag_masks)

                elif not isinstance(flag_masks, (list, tuple)):
                    self._ds[var_name].attrs['flag_masks'] = [flag_masks]

            except KeyError:
                pass
