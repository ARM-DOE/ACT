import numpy as np
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
        sever having higher integer numbers.

        cleanup_qc : boolean
            Call clean.cleanup() method to convert to standardized ancillary quality control
            variables. The quality control summary requires the current embedded quality
            control variables to use ACT standards.

        Returns
        -------
        return_ds : xarray.Dataset
            ACT Xarray dataset with quality control variales converted to summary flag values.

        """

        standard_assessments = [
            'Suspect',
            'Indeterminate',
            'Incorrect',
            'Bad',]
        standard_meanings = [
            "Data suspect, further analysis recommended",
            "Data suspect, further analysis recommended",
            "Data incorrect, use not recommended",
            "Data incorrect, use not recommended",]

        return_ds = self._ds.copy()

        if cleanup_qc:
            self._ds.clean.cleanup()

        added = False
        for var_name in list(self._ds.data_vars):
            qc_var_name = self.check_for_ancillary_qc(var_name, add_if_missing=False, cleanup=False)

            if qc_var_name is None:
                continue

            added = True

            assessments = list(set(self._ds[qc_var_name].attrs['flag_assessments']))
            del return_ds[qc_var_name]

            return_ds.qcfilter.add_test(
                var_name,
                index=None,
                test_number=0,
                test_meaning='Passing all quality control tests',
                test_assessment='Passing',
                flag_value=True,)

            for ii, assessment in enumerate(standard_assessments):
                if assessment not in assessments:
                    continue

                qc_ma = self.get_masked_data(var_name, rm_assessments=assessment)

                # Do not really know how to handle scalars yet.
                if len(qc_ma.mask.shape) == 0:
                    continue

                return_ds.qcfilter.add_test(
                    var_name,
                    index=np.where(qc_ma.mask),
                    test_meaning=standard_meanings[ii],
                    test_assessment=assessment,
                    flag_value=True)

        if added:
            history = return_ds.attrs['history']
            history += (
                " ; Quality control summary implemented by ACT at "
                f"{datetime.datetime.utcnow().isoformat()} UTC.")
            return_ds.attrs['history'] = history

        return return_ds
