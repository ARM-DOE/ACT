import numpy as np
import warnings
from act.utils.data_utils import convert_units
from act.utils.datetime_utils import determine_time_delta
import copy
import xarray as xr


class QCTests:

    def compare_time_series_trends(self, var_name=None, comp_dataset=None, comp_var_name=None,
                                   time_match_threshhold=60, time_shift=60*60, time_step=None,
                                   time_qc_threshold=60*15):
        """
        docstring

        """

        # If no comparison variable name given assume matches variable name
        if comp_var_name is None:
            comp_var_name = var_name

        # If no comparison Dataset given assume self Dataset
        if comp_dataset is None:
            comp_dataset = self

        # Extract copy of DataArray for work below
        self_da = copy.deepcopy(self._obj[var_name])
        comp_da = copy.deepcopy(comp_dataset[comp_var_name])

        # Convert comp data units to match
        comp_da.values = convert_units(comp_da.values, comp_da.attrs['units'],
                                       self_da.attrs['units'])
        comp_da.attrs['units'] = self_da.attrs['units']

        # self_da_shifted = self_da.reindex(
        #         time=comp_da.time.values, method='nearest',
        #         tolerance=np.timedelta64(time_match_threshhold, 's'))
        # avg_diff = np.nanmean(self_da_shifted.values - comp_da.values)

        # Match comparison data to time of data
        if time_step is None:
            time_step = determine_time_delta(self._obj['time'].values)
        sum_diff = np.array([], dtype=float)
        time_diff = np.array([], dtype=np.int32)
        for tm_shift in range(-1*time_shift, time_shift + int(time_step), int(time_step)):
            self_da_shifted = self_da.assign_coords(
                time = self_da.time.values.astype('datetime64[s]') + tm_shift)

            data_matched, comp_data_matched = xr.align(self_da, comp_da)
            self_da_shifted = self_da_shifted.reindex(
                time=comp_da.time.values, method='nearest',
                tolerance=np.timedelta64(time_match_threshhold, 's'))
            diff = np.abs(self_da_shifted.values - comp_da.values)
            sum_diff = np.append(sum_diff, np.nansum(diff))
            time_diff = np.append(time_diff, tm_shift)

        index = np.argmin(np.abs(sum_diff))
        time_diff = time_diff[index]

        index = None
        if np.abs(time_diff) > time_qc_threshold:
            index=np.arange(0, self_da.size)
        meaning = (f"Time shift detected with Minimum Difference test. Comparison of "
                   f"{var_name} with {comp_var_name} off by {time_diff} seconds "
                   f"exceeding absolute threshold of {time_qc_threshold} seconds.")
        self._obj.qcfilter.add_test(var_name, index=index,
                                    test_meaning=meaning, test_assessment='Indeterminate')


    def add_pdf_test(self, var_name, previous_time, previous_data, xnumbin=None, ynumbin=None,
                     threshold=None, ybinsize=None, ymin=None, ymax=None, test_number=None,
                     exclude_index=None, test_assessment='Indeterminate',
                     test_meaning=None, flag_value=False, prepend_text=None):
        """
        docstring

        """

        print(var_name)

        # Find the min and max of the variable for calculating limits--;

        # Calculate the number of bins sizes
        if xnumbin is None:
            xnumbin = np.timedelta64(days, 'D')
        if ynumbin is None:
            ynumbin = (ymax - ymin) / ybinsize

        if ymin is None:
            ymin=np.floor(np.nanmin(previous_data))
        if ymax is None:
            ymax=np.nanmax(previous_data)

        if ymax < ymin:
           ymax=np.nanmax(previous_data)
           ymin=np.nanmin(previous_data)

        # Make sure that the limits are not the same
        if ymin == ymax:
            ymax += 1.

        # Get the x-axis min and max
        xmax = np.max(previous_time)
        xmin = np.min(previous_time)

        # Calculate the number of bins sizes
        if xnumbin is None:
            xnumbin = np.timedelta64(days, 'D')
        if ynumbin is None:
            ynumbin = (ymax - ymin) / binsize

        # Calculate the 2d Histogram
        time_dtype = 'datetime64[s]'
        result, xedges, yedges = np.histogram2d(
            new_object['time'].values.astype(time_dtype).astype(int), var,
            bins=[xnumbin.astype(int), ynumbin],
            range=[[xmin.astype(time_dtype).astype(int), xmax.astype(time_dtype).astype(int)],
                   [ymin, ymax]])
#        xedges = xedges.astype(time_dtype)


