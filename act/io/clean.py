import xarray as xr
import re
import numpy as np
import copy

@xr.register_dataset_accessor('clean')
class CleanDataset(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def matched_qc_variables(self):
        '''Find variables that are QC variables and return list of names.

        Returns
        -------
        variables: list
            A list of strings containing the name of each variable.
        '''

        variables = []

        # Will need to find all historical cases and add to list
        qc_dict = {'description':
                   ["See global attributes for individual bit descriptions.",
                    "This field contains bit packed integer values, where "
                    "each bit represents a QC test on the data. Non-zero "
                    "bits indicate the QC condition given in the description "
                    "for those bits; a value of 0 (no bits set) indicates "
                    "the data has not failed any QC tests."
                    ]
                   }

        # Loop over each variable and look for a match to an attribute that
        # would exist if the variable is a QC variable
        for var in self._obj.data_vars:
            attributes = self._obj[var].attrs
            for att_name in attributes:
                if att_name in qc_dict.keys():
                    for value in qc_dict[att_name]:
                        if attributes[att_name] == value:
                            variables.append(var)
                            break

        return variables

    def cleanup(self, cleanup_arm_qc=True, clean_arm_state_vars=None,
                handle_missing_value=True, link_qc_variables=True,
                add_source_files=True, **kwargs):
        '''
        Wrapper method to automatically call all the standard methods
        for obj cleanup. Has options to pass through keywords.

        Parameters
        ----------
        cleanup_arm_qc : bool
            Option to clean xarray object from ARM QC to CF QC standards
            Default is True
        clean_arm_state_vars : list of str
            Option to clean xarray object state variables from ARM to CF
            standards.
            Pass in list of variable names. Default is None.
        handle_missing_value : bool
            Option to update and fix missing values. Default is True
        link_qc_variables : bool
            Option to link QC variablers through ancillary_variables if not
            already set. Default is True
        add_source_files : bool
            Option to add global attribute to list files read.
            Default is True.

        '''

        # Clean up ARM QC to be more like CF state fields
        if cleanup_arm_qc:
            self._obj.clean.clean_arm_qc(**kwargs)

        # Clean up ARM state fields to be more liek CF state fields
        if clean_arm_state_vars:
            self._obj.clean.clean_arm_state_variables(clean_arm_state_vars)

        # Correctly clean up missing value indicators
        if handle_missing_value:
            self._obj.clean.handle_missing_values(**kwargs)

        # Update to add some ancillary_variables linkages
        # between variable and QC variable
        if link_qc_variables:
            self._obj.clean.link_variables()

    def handle_missing_values(self, **kwargs):
        '''Correctly handle missing_value and _FillValue in object.

        Parameters
        ----------
        additional_missing_values : list
            List of values to also use as missing value indicator. These
            will be checked and set to the one true missing value.

        Returns
        -------
        object : xarray dataset
            xarray dataset object of read data or None if no data found.
            If the variable does not have a missing_value or _FillValue
            will use -9999 set to correct type. Skips ARM QC variables
            and variables with flag_meanings attributes.

        Todo
        ----
            Need to give option of choosing missing_value or _FillValue
            as the attribute to use. Currently converts everything to
            missing_value.

        '''

        additional_missing_values = None
        if 'additional_missing_values' in kwargs.keys():
            additional_missing_values = kwargs[
                'additional_missing_values=None']

        attr_names = ['missing_value', '_FillValue']
        default_missing_value = np.int32(-9999)
        for var in self._obj.data_vars:
            # Skip data quality fields.
            if('Quality check results on field:' in
                    self._obj[var].attrs['long_name']):
                continue

            # Skip state fields.
            try:
                self._obj[var].attrs['flag_meanings']
                continue
            except KeyError:
                pass

            data_type = self._obj[var].values.dtype
            missing_values = []
            if additional_missing_values:
                missing_values = [additional_missing_values]
            for attr_name in attr_names:
                try:
                    missing_value = self._obj[var].attrs[attr_name]
                    if isinstance(missing_value, (list, tuple)):
                        missing_values.extend(missing_value)
                    else:
                        missing_values.append(missing_value)
                except KeyError:
                    continue

            if not missing_values:
                missing_values = [default_missing_value.astype(data_type)]

            try:
                for ii, missing_value in enumerate(missing_values):
                    missing_values[ii] = missing_value.astype(data_type)

                if len(missing_values) > 1:
                    data = self._obj[var].values
                    for ii, missing_value in enumerate(missing_values):
                        if ii == 0:
                            continue
                        data[data == missing_value] = missing_values[0]
                    self._obj[var].values = data

                self._obj[var].attrs['missing_value'] = copy.copy(
                    missing_values[0])
            except AttributeError:
                pass

    def get_attr_info(self, variable=None, flag=False):
        '''Get QC definitions and return as dictionary.

        Parameters
        ----------
        variable : str
            Variable name to get attribute information.
        flag : bool
            Optional flag indicating if QC is expected to be bitpacked
            or integer. Flag = True indicates integer QC. Default
            is bitpacked or False.

        Returns
        -------
        attributes dictionary : dict
            A dictionary contianing the attribute information converted from
            ARM QC to CF QC. Keys in dict will be set only if values to
            populate. All keys include 'flag_meanings', 'flag_masks',
            'flag_values', 'flag_assessments', 'flag_tests', 'arm_attributes'.

        '''

        string = 'bit'
        if flag:
            string = 'flag'

        try:
            if variable:
                attr_description_pattern = (r"(^" + string +
                                            r")_([0-9])_(description$)")
                attr_assessment_pattern = (r"(^" + string +
                                           r")_([0-9])_(assessment$)")
                attributes = self._obj[variable].attrs
            else:
                attr_description_pattern = (r"(^qc_" + string +
                                            r")_([0-9])_(description$)")
                attr_assessment_pattern = (r"(^qc_" + string +
                                           r")_([0-9])_(assessment$)")
                attributes = self._obj.attrs
        except KeyError:
            return None

        assessment_bit_num = []
        description_bit_num = []
        flag_masks = []
        flag_meanings = []
        flag_assessments = []
        arm_attributes = []
        for att_name in attributes:
            description = re.match(attr_description_pattern, att_name)
            try:
                description_bit_num.append(int(description.groups()[1]))
                flag_meanings.append(attributes[att_name])
                arm_attributes.append(att_name)
            except AttributeError:
                pass

            assessment = re.match(attr_assessment_pattern, att_name)
            try:
                assessment_bit_num.append(int(assessment.groups()[1]))
                flag_assessments.append(attributes[att_name])
                arm_attributes.append(att_name)
            except AttributeError:
                pass

        # Sort on bit number to ensure correct description order
        index = np.argsort(description_bit_num)
        flag_meanings = np.array(flag_meanings)
        description_bit_num = np.array(description_bit_num)
        flag_meanings = list(flag_meanings[index])
        description_bit_num = list(description_bit_num[index])

        # Sort on bit number to ensure correct assessment order
        index = np.argsort(assessment_bit_num)
        flag_assessments = np.array(flag_assessments)
        flag_assessments = list(flag_assessments[index])

        # Convert bit number to mask number
        if description_bit_num:
            flag_masks = np.array(description_bit_num)
            flag_masks = list(np.left_shift(1, flag_masks - 1))

        # build dictionary to return values
        return_dict = dict()
        if flag_meanings:
            return_dict['flag_meanings'] = flag_meanings
        if not flag and flag_masks:
            return_dict['flag_masks'] = flag_masks
        if flag and flag_masks:
            return_dict['flag_values'] = description_bit_num
        if flag_assessments:
            return_dict['flag_assessments'] = flag_assessments
        if description_bit_num:
            return_dict['flag_tests'] = description_bit_num
        if arm_attributes:
            return_dict['arm_attributes'] = arm_attributes

        # if nothing to return set to None
        if not return_dict:
            return_dict = None

        return return_dict

    def clean_arm_state_variables(self, variables, **kwargs):
        '''
        Function to clean up state variables to use more CF method.

        Parameters
        ----------
        variables : list of str
            List of variable names to update.

        override_cf_flag : bool
            Option to overwrite CF flag_meanings attribute if it exists
            with the values from ARM QC bit_#_description. Default is True.
        clean_units_string : bool
            Option to update units string if set to 'unitless' to be
            udunits '1'. Default is True.
        integer_flag : bool
            Passthrogh keyword of 'flag' for get_attr_info()

        '''

        override_cf_flag = True
        if 'override_cf_flag' in kwargs.keys():
            override_cf_flag = kwargs['override_cf_flag']

        clean_units_string = True
        if 'clean_units_string' in kwargs.keys():
            clean_units_string = kwargs['clean_units_string']

        integer_flag = True
        if 'integer_flag' in kwargs.keys():
            integer_flag = kwargs['integer_flag']

        if isinstance(variables, str):
            variables = [copy.copy(variables)]

        for var in variables:
            flag_info = self.get_attr_info(variable=var, flag=integer_flag)
            if not flag_info:
                return

            # Add new attributes to variable
            for attr in ['flag_values', 'flag_meanings', 'flag_masks']:
                attr_exists = attr in self._obj[var].attrs.keys()

                try:
                    if not attr_exists:
                        self._obj[var].attrs[attr] = flag_info[attr]
                    elif override_cf_flag:
                        self._obj[var].attrs[attr] = flag_info[attr]
                except KeyError:
                    pass

            # Remove replaced attributes
            arm_attributes = flag_info['arm_attributes']
            for attr in arm_attributes:
                try:
                    del self._obj[var].attrs[attr]
                except KeyError:
                    pass

            # Clean up units attribute from unitless to udunits '1'
            if (clean_units_string and
                    self._obj[var].attrs['units'] == 'unitless'):
                self._obj[var].attrs['units'] = '1'

    def correct_valid_minmax(self, qc_variable):
        '''Function to correct the name and location of
           QC limit variables that use
           valid_min and valid_max incorrectly.
        '''

        test_dict = {'valid_min': 'fail_min',
                     'valid_max': 'fail_max',
                     'valid_delta': 'fail_delta'}

        aa = re.match(r"^qc_(.+)", qc_variable)
        variable = None
        try:
            variable = aa.groups()[0]
        except AttributeError:
            return

        made_change = False
        flag_meanings = copy.copy(
            self._obj[qc_variable].attrs['flag_meanings'])
        for attr in test_dict.keys():
            for ii, test in enumerate(flag_meanings):
                if attr in test:
                    flag_meanings[ii] = re.sub(attr, test_dict[attr], test)
                    made_change = True
                    try:
                        self._obj[qc_variable].attrs[test_dict[attr]] = \
                            copy.copy(self._obj[variable].attrs[attr])
                        del self._obj[variable].attrs[attr]
                    except KeyError:
                        pass

        if made_change:
            self._obj[qc_variable].attrs['flag_meanings'] = flag_meanings

    def link_variables(self):
        '''Add some attributes to link and explain data
           to QC data relationship.'''

        for var in self._obj.data_vars:
            aa = re.match(r"^qc_(.+)", var)
            try:
                variable = aa.groups()[0]
                qc_variable = var
            except AttributeError:
                continue
            # Skip data quality fields.
            if not ('Quality check results on field:' in
                    self._obj[var].attrs['long_name']):
                continue

            # Get existing data variable ancillary_variables attribute
            try:
                ancillary_variables = self._obj[variable].\
                    attrs['ancillary_variables']
            except KeyError:
                ancillary_variables = ''

            # If the QC variable is not in ancillary_variables add
            if qc_variable not in ancillary_variables:
                ancillary_variables = qc_variable
            self._obj[variable].attrs['ancillary_variables']\
                = ancillary_variables

            # Get the standard name from QC variable
            try:
                qc_var_standard_name = self._obj[qc_variable].\
                    attrs['standard_name']
            except KeyError:
                qc_var_standard_name = None

            # Add quality_flag to standard name
            if qc_var_standard_name:
                qc_var_standard_name = ' '.join([qc_var_standard_name,
                                                'quality_flag'])
            else:
                qc_var_standard_name = 'quality_flag'

            # put standard_name in QC variable obj
            self._obj[qc_variable].attrs['standard_name']\
                = qc_var_standard_name

    def get_qc_attr_info(self, variable=None, flag=False):
        '''
        Get QC definitions and return as dictionary.

        Parameters
        ----------
        variable : str
            Variable name to return QC attribute information. Default is None
            which returns global attribute QC information.
        flag : bool
            Option to indicate type of QC, bitpacked or integer. Flag set to
            True indicates integer. Default is False or bitpacked.

        Todo
        ----
            Need to update to handle bitpacked vs. integer automatically.

        '''

        string = 'bit'
        if flag:
            string = 'flag'

        try:
            if variable:
                attr_description_pattern = (r"(^" + string +
                                            r")_([0-9])_(description$)")
                attr_assessment_pattern = (r"(^" + string +
                                           r")_([0-9])_(assessment$)")
                attributes = self._obj[variable].attrs
            else:
                attr_description_pattern = (r"(^qc_" + string +
                                            r")_([0-9])_(description$)")
                attr_assessment_pattern = (r"(^qc_" + string +
                                           r")_([0-9])_(assessment$)")
                attributes = self._obj.attrs
        except KeyError:
            return None

        assessment_bit_num = []
        description_bit_num = []
        flag_masks = []
        flag_meanings = []
        flag_assessments = []
        arm_attributes = []
        for att_name in attributes:
            description = re.match(attr_description_pattern, att_name)
            try:
                description_bit_num.append(int(description.groups()[1]))
                flag_meanings.append(attributes[att_name])
                arm_attributes.append(att_name)
            except AttributeError:
                pass

            assessment = re.match(attr_assessment_pattern, att_name)
            try:
                assessment_bit_num.append(int(assessment.groups()[1]))
                flag_assessments.append(attributes[att_name])
                arm_attributes.append(att_name)
            except AttributeError:
                pass

        # Sort on bit number to ensure correct description order
        index = np.argsort(description_bit_num)
        flag_meanings = np.array(flag_meanings)
        description_bit_num = np.array(description_bit_num)
        flag_meanings = list(flag_meanings[index])
        description_bit_num = list(description_bit_num[index])

        # Sort on bit number to ensure correct assessment order
        if len(flag_assessments) > 0:
            index = np.argsort(assessment_bit_num)
            flag_assessments = np.array(flag_assessments)
            flag_assessments = list(flag_assessments[index])
        else:
            flag_assessments = ['Unknown'] * len(flag_meanings)

        # Convert bit number to mask number
        if description_bit_num:
            flag_masks = np.array(description_bit_num)
            flag_masks = list(np.left_shift(1, flag_masks - 1))

        # build dictionary to return values
        return_dict = dict()
        if flag_meanings:
            return_dict['flag_meanings'] = flag_meanings
        if not flag and flag_masks:
            return_dict['flag_masks'] = flag_masks
        if flag and flag_masks:
            return_dict['flag_values'] = description_bit_num
        if flag_assessments:
            return_dict['flag_assessments'] = flag_assessments
        if description_bit_num:
            return_dict['flag_tests'] = description_bit_num
        if arm_attributes:
            return_dict['arm_attributes'] = arm_attributes

        # if nothing to return set to None
        if not return_dict:
            return_dict = None

        return return_dict

    def clean_arm_qc(self, **kwargs):
        """Function to clean up xarray object QC variables.

        Parameters
        ----------
        override_cf_flag : bool
            Option to overwrite CF flag_masks, flag_meanings, flag_values
            if exists. Default is True.
        clean_units_string : bool
            Option to clean up units string from 'unitless' to udunits '1'.
            Default is True.
        correct_valid_min_max : bool
            Option to correct use of valid_min and valid_max with QC variables
            by moving from data variable to QC varible, renaming to fail_min,
            fail_max and fail_detla if the valid_min or valid_max is listed
            in bit discription attribute. If not listed as used with QC will
            assume is being used correctly. Default is True.
        """

        override_cf_flag = True
        if 'override_cf_flag' in kwargs.keys():
            override_cf_flag = kwargs['override_cf_flag']

        clean_units_string = True
        if 'clean_units_string' in kwargs.keys():
            clean_units_string = kwargs['clean_units_string']

        correct_valid_min_max = True
        if 'correct_valid_min_max' in kwargs.keys():
            correct_valid_min_max = kwargs['correct_valid_min_max']

        global_qc = self.get_qc_attr_info()

        var_num = -1
        for qc_var in self.matched_qc_variables:
            var_num += 1
            # if first pass try to clean up global attributes
            if var_num == 0 and global_qc != None:
                global_attributes = global_qc['arm_attributes']
                global_attributes.extend(['qc_bit_comment'])
                for attr in global_attributes:
                    try:
                        del self._obj.attrs[attr]
                    except KeyError:
                        pass

            # Clean up units attribute from unitless to udunits '1'
            if (clean_units_string and
                    self._obj[qc_var].attrs['units'] == 'unitless'):
                self._obj[qc_var].attrs['units'] = '1'

            qc_attributes = self.get_qc_attr_info(variable=qc_var)
            if not qc_attributes:
                qc_attributes = global_qc

            # Add new attributes to variable
            for attr in ['flag_masks', 'flag_meanings', 'flag_assessments']:
                attr_exists = attr in self._obj[qc_var].attrs.keys()

                if not attr_exists:
                    self._obj[qc_var].attrs[attr] = qc_attributes[attr]
                elif override_cf_flag:
                    self._obj[qc_var].attrs[attr] = qc_attributes[attr]

            # Remove replaced attributes
            arm_attributes = qc_attributes['arm_attributes']
            arm_attributes.extend(['description', 'flag_method'])
            for attr in arm_attributes:
                try:
                    del self._obj[qc_var].attrs[attr]
                except KeyError:
                    pass

            # Check for use of valid_min and valid_max as QC limits and fix
            if correct_valid_min_max:
                self._obj.clean.correct_valid_minmax(qc_var)
