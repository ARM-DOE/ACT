"""
Functions and methods for performing comparison tests.

"""
import copy

import numpy as np
import xarray as xr

from act.utils.data_utils import convert_units
from act.utils.datetime_utils import determine_time_delta


class QCTests:
    def compare_time_series_trends(
        self,
        var_name=None,
        comp_dataset=None,
        comp_var_name=None,
        time_match_threshhold=60,
        time_shift=60 * 60,
        time_step=None,
        time_qc_threshold=60 * 15,
    ):
        """
        Method to perform a time series comparison test between two Xarray Datasets
        to detect a shift in time based on two similar variables. This test will
        compare two similar measurements and look to see if there is a time shift
        forwards or backwards that makes the comparison better. If so assume the
        time has shifted.

        This test is not 100% accurate. It may be fooled with noisy data. Use
        with your own discretion.

        Parameters
        ----------
        var_name : str
            Data variable name.
        comp_dataset : Xarray Dataset
            Dataset containing comparison data to use in test.
        comp_var_name : str
            Name of variable in comp_dataset to use in test.
        time_match_threshhold : int
            Number of seconds to use in tolerance with reindex() method
            to match time from self to comparison Dataset.
        time_shift : int
            Number of seconds to shift analysis window before and after
            the time in self Dataset time.
        time_step : int
            Time step in seconds for self Dataset time. If not provided
            will attempt to find the most common time step.
        time_qc_threshold : int
            The quality control threshold to use for setting test. If the
            calculated time shift is larger than this value will set all
            values in the QC variable to a tripped test value.

        Returns
        -------
        test_info : tuple
            A tuple containing test information including var_name, qc variable name,
            test_number, test_meaning, test_assessment

        """
        # If no comparison variable name given assume matches variable name
        if comp_var_name is None:
            comp_var_name = var_name

        # If no comparison Dataset given assume self Dataset
        if comp_dataset is None:
            comp_dataset = self

        # Extract copy of DataArray for work below
        self_da = copy.deepcopy(self._ds[var_name])
        comp_da = copy.deepcopy(comp_dataset[comp_var_name])

        # Convert comp data units to match
        comp_da.values = convert_units(
            comp_da.values, comp_da.attrs['units'], self_da.attrs['units']
        )
        comp_da.attrs['units'] = self_da.attrs['units']

        # Match comparison data to time of data
        if time_step is None:
            time_step = determine_time_delta(self._ds['time'].values)
        sum_diff = np.array([], dtype=float)
        time_diff = np.array([], dtype=np.int32)
        for tm_shift in range(-1 * time_shift, time_shift + int(time_step), int(time_step)):
            self_da_shifted = self_da.assign_coords(
                time=self_da.time.values.astype('datetime64[s]') + tm_shift
            )

            data_matched, comp_data_matched = xr.align(self_da, comp_da)
            self_da_shifted = self_da_shifted.reindex(
                time=comp_da.time.values,
                method='nearest',
                tolerance=np.timedelta64(time_match_threshhold, 's'),
            )
            diff = np.abs(self_da_shifted.values - comp_da.values)
            sum_diff = np.append(sum_diff, np.nansum(diff))
            time_diff = np.append(time_diff, tm_shift)

        index = np.argmin(np.abs(sum_diff))
        time_diff = time_diff[index]

        index = None
        if np.abs(time_diff) > time_qc_threshold:
            index = np.arange(0, self_da.size)
        meaning = (
            f'Time shift detected with Minimum Difference test. Comparison of '
            f'{var_name} with {comp_var_name} off by {time_diff} seconds '
            f'exceeding absolute threshold of {time_qc_threshold} seconds.'
        )
        result = self._ds.qcfilter.add_test(
            var_name, index=index, test_meaning=meaning, test_assessment='Indeterminate'
        )

        return result
