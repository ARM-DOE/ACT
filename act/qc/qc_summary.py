"""
Method for creating Quality Control Summary variables from the embedded
quality control varialbes. The summary variable is a simplified version of
quality control that uses flag integers instead of bit-packed masks. The
number of descriptions is simplified to consolidate all categories into one
description.

"""

import datetime


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

    def create_qc_summary(self, cleanup_qc=False):
        """
        Method to convert embedded quality control to summary QC that utilzes
        flag values instead of flag masks and summarizes the assessments to only
        a few states. Lowest level of quality control will be listed first with most
        sever having higher integer numbers. Dataset is updated in place.

        cleanup_qc : boolean
            Call clean.cleanup() method to convert to standardized ancillary quality control
            variables. The quality control summary requires the current embedded quality
            control variables to use ACT standards.

        Returns
        -------
        return_ds : Xarray.dataset
            ACT Xarray dataset with quality control variables converted to summary flag values.

        """

        standard_assessments = [
            'Suspect',
            'Indeterminate',
            'Incorrect',
            'Bad',
        ]
        standard_meanings = [
            "Data suspect, further analysis recommended",
            "Data suspect, further analysis recommended",
            "Data incorrect, use not recommended",
            "Data incorrect, use not recommended",
        ]

        if cleanup_qc:
            self._ds.clean.cleanup()

        return_ds = self._ds.copy()

        added = False
        for var_name in list(self._ds.data_vars):
            qc_var_name = self.check_for_ancillary_qc(var_name, add_if_missing=False, cleanup=False)

            if qc_var_name is None:
                continue

            added = True

            assessments = list(set(self._ds[qc_var_name].attrs['flag_assessments']))

            import xarray as xr

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

            for ii, assessment in enumerate(standard_assessments):
                if assessment not in assessments:
                    continue

                qc_mask = self.get_masked_data(
                    var_name, rm_assessments=assessment, return_mask_only=True
                )

                # Do not really know how to handle scalars yet.
                if qc_mask.ndim == 0:
                    continue

                return_ds.qcfilter.add_test(
                    var_name,
                    index=qc_mask,
                    test_meaning=standard_meanings[ii],
                    test_assessment=assessment,
                    flag_value=True,
                )

            self._ds.update({qc_var_name: return_ds[qc_var_name]})

        if added:
            history = return_ds.attrs['history']
            history += (
                " ; Quality control summary implemented by ACT at "
                f"{datetime.datetime.utcnow().isoformat()} UTC."
            )
            return_ds.attrs['history'] = history

        return return_ds
