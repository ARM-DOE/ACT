"""
Method for creating Quality Control Summary variables from the embedded
quality control varialbes. The summary variable is a simplified version of
quality control that uses flag integers instead of bit-packed masks. The
number of descriptions is simplified to consolidate all categories into one
description.

"""

import datetime
import copy
import xarray as xr
import warnings


class QCSummary:
    """
    This is a Mixins class used to allow using qcfilter class that is already
    registered to the Xarray dataset. All the methods in this class will be added
    to the qcfilter class. Doing this to make the code spread across more files
    so it is more manageable and readable.

    """

    def __init__(self, ds):
        """initialize"""
        self._ds = ds

    def create_qc_summary(
        self,
        cleanup_qc=False,
        remove_attrs=['fail_min', 'fail_max', 'fail_delta'],
        normalize_assessment=True,
    ):
        """
        Method to convert embedded quality control to summary QC that utilzes
        flag values instead of flag masks and summarizes the assessments to only
        a few states. Lowest level of quality control will be listed first with most
        sever having higher integer numbers. Dataset is updated in place.

        cleanup_qc : boolean
            Call clean.cleanup() method to convert to standardized ancillary quality control
            variables. The quality control summary requires the current embedded quality
            control variables to use ACT standards.
        remove_attrs : None, list
            Quality Control variable attributes to remove after creating the summary.
        normalize_assessment : bool
            Option to clean up assessments to use the same terminology.


        Returns
        -------
        return_ds : Xarray.dataset
            ACT Xarray dataset with quality control variables converted to summary flag values.

        """

        standard_meanings = {
            'Suspect': "Data suspect further analysis recommended",
            'Indeterminate': "Data suspect further analysis recommended",
            'Incorrect': "Data incorrect use not recommended",
            'Bad': "Data incorrect use not recommended",
        }

        if cleanup_qc:
            self._ds.clean.cleanup()

        if normalize_assessment:
            self._ds.clean.normalize_assessment()

        return_ds = self._ds.copy()

        added = False
        for var_name in list(self._ds.data_vars):
            qc_var_name = self.check_for_ancillary_qc(var_name, add_if_missing=False, cleanup=False)

            if qc_var_name is None:
                continue

            # Do not really know how to handle scalars yet.
            if return_ds[qc_var_name].ndim == 0:
                warnings.warn(
                    f'Unable to process scalar variable {var_name}. '
                    'Scalar variables currently not implemented.'
                )
                continue

            added = True

            result = xr.zeros_like(return_ds[qc_var_name])
            for attr in ['flag_masks', 'flag_meanings', 'flag_assessments', 'flag_values']:
                try:
                    del result.attrs[attr]
                except KeyError:
                    pass

            return_ds[qc_var_name] = result

            return_ds.qcfilter.add_test(
                var_name,
                index=None,
                test_number=0,
                test_meaning='Not failing quality control tests',
                test_assessment='Not failing',
                flag_value=True,
            )

            flag_assessments = list(standard_meanings.keys())
            added_assessments = set(self._ds[qc_var_name].attrs['flag_assessments']) - set(
                flag_assessments
            )
            flag_assessments += list(added_assessments)
            for ii, assessment in enumerate(flag_assessments):
                try:
                    standard_meaning = standard_meanings[assessment.capitalize()]
                except KeyError:
                    standard_meaning = f"Data {assessment}"

                qc_mask = self.get_masked_data(
                    var_name, rm_assessments=assessment, return_mask_only=True
                )

                # # Do not really know how to handle scalars yet.

                return_ds.qcfilter.add_test(
                    var_name,
                    index=qc_mask,
                    test_meaning=standard_meaning,
                    test_assessment=assessment,
                    flag_value=True,
                )

            # Remove fail limit variable attributes
            if remove_attrs is not None:
                for att_name in copy.copy(list(return_ds[qc_var_name].attrs.keys())):
                    if att_name in remove_attrs:
                        del return_ds[qc_var_name].attrs[att_name]

            self._ds.update({qc_var_name: return_ds[qc_var_name]})

        if added:
            from act import __version__ as version

            history_value = (
                f"Quality control summary implemented by ACT-{version} at "
                f"{datetime.datetime.utcnow().replace(microsecond=0)} UTC"
            )

            if 'history' in list(return_ds.attrs.keys()):
                return_ds.attrs['history'] += f" ; {history_value}"
            else:
                return_ds.attrs['history'] = history_value

        return return_ds
