"""
act.qc.filter
=====================
Functions for creating ancillary quality control variables and filters (masks)
which can be used it various corrections routines in ACT.
"""

import numpy as np
import xarray as xr


@xr.register_dataset_accessor('qcfilter')
class QCFilter(object):
    """
    A class for building a boolean arrays for filtering data based on
    a set of condition typically based on the values in the data fields.
    These filter can be used in various algorithms and calculations within
    ACT.

    Attributes
    ----------
    check_for_ancillary_qc
        Method to check if a quality control variable exist in the dataset.
    create_qc_variable
        Method to create a new quality control variable in the dataset
        and set default values of 0.
    update_ancillary_variable
        Method to check if variable attribute ancillary_variables
        is set and if not to set to correct quality control variable name.

    """

    def __init__(self, xarray_obj):
        """ initialize """
        self._obj = xarray_obj

    def check_for_ancillary_qc(self, var_name):
        '''
        Method to check for corresponding quality control variable in
        the dataset. If it does not exist will create and correctly
        link the data variable to quality control variable with
        variable ancillary_variables attribute.

        Parameters
        ----------
        var_name : str
            data variable name

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

    def create_qc_variable(self, var_name):
        '''
        Method to create a quality control variable in the dataset.
        Will destroy a variable with the same name as the new quality
        control variable name that this modules creates. Will try not
        to destroy the qc variable by appending numbers to the variable
        name if needed.

        Parameters
        ----------
        var_name : str
            data variable name

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
                                  np.zeros(variable.shape))
        self._obj[qc_var_name].attrs['long_name'] = qc_variable_long_name
        self._obj[qc_var_name].attrs['units'] = '1'
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
