import yaml
import numpy as np
from pathlib import Path
from dateutil import parser

#   Example of the YAML file and how to construct.
#   The times are set as inclusive start to inclusive end time.
#   Different formats are acceptable as displayed with temp_mean
#   Good example below.
#
#   The general format is to use the variable name as initial key
#   followed by an assessment key, followed by a 'description' key which
#   is a description of that test. Each test listed will insert a new test
#   into the ancillary quality control variable. After the 'description'
#   list all time ranges as a YAML array.
#   If a test is to be applied to all variables list under the specail
#   variable name '_all'.
#
#   The file should be named same as datastream name with standard
#   YAML file extension when only providing the directoy.
#   For example the file below would be named sgpmetE13.b1.yaml
#
#
#  _all:
#    Bad:
#      Values are bad for all:
#        - 2020-01-21 01:01:02, 2020-01-21 01:03:13
#        - 2020-01-21 02:01:02, 2020-01-21 04:03:13
#   Suspect:
#     Values are suspect for all: []
#
# temp_mean:
#   Bad:
#     Values are bad:
#       - 2020-01-01 00:01:02, 2020-01-01 00:03:44
#       - 2020-01-02 01:01:02, 2020-01-02 01:03:13
#       - 2020-02-01 00:01:02, 2020-02-01 00:03:44
#       - 2020-03-02 01:01:02, 2020-03-02 01:03:13
#       - 2020-01-21 01:01:02, 2020-01-21 01:03:13
#   Suspect:
#     Values are suspect:
#       - 2020-01-01 02:04:02, 2020-01-01 02:05:44
#   Good:
#     Values are good:
#       - 2020-01-01 00:08:02, 2020-01-01 00:09:44
#       - Jan 1, 2020 00:08:02 ; January 1, 2020 00:09:44 AM
#       - 2020-01-01 00:08 | 2020-01-01 00:09
#       - 2020-01-01T00:08:02 ; 2020-01-01T00:09:44
#
# rh_mean:
#   Bad:
#     Values are bad:
#       - 2020-01-01 00:01:02, 2020-01-01 00:03:44
#       - 2020-01-02 00:01:02, 2020-01-02 00:03:44
#   Suspect:
#     Values are suspect:
#       - 2020-01-01 00:04:02, 2020-01-01 00:05:44


def read_yaml_supplemental_qc(
    ds,
    fullpath,
    variables=None,
    assessments=None,
    datetime64=True,
    time_delim=(';', ',', '|', r'\t'),
    none_if_empty=True,
    quiet=False,
):
    """
    Returns a dictionary converstion of YAML file for flagging data. The dictionary
    will contain variable names as first key, assessents as second keys containing
    test description as third key with time as last key. Multiple descriptions
    are allowed.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset containing data.
    fullpath : str or `pathlib.Path`
        Full path to file to read or directory containing YAML files to read. If providing
        a directory a file with the datastream name ending in .yaml or .yml must exists.
    variables : str, list of str or None
        Optional, variable names to keep in dictionary. All others will be removed
        before returning dictionary. If no variables are left will return None.
    assessments : str, list of str or None
        Optional, assessments categories to keep in dictionary. All others will be removed
        before returning dictionary.
    datetime64 : boolean
        Convert the string value list to 2D numpy datetime64 array with first value
        the start time and second value end time. If the time list in the YAML file
        is empty, the 'time' numpy array will be a single value instead of 2D array
        and set to numpy datetime64 NaT.
    time_delim : str or list of str or tuple of str
        Optional, character delimiter to use when parsing times.
    none_if_empty : boolean
        Return None instead of empty dictionary
    quiet : boolean
        Suppress information about not finding a YAML file to read.

    Returns
    -------
        Dictionary of [variable names][assessments][description] and time values
        or if the dictionary is empty after processing options and none_if_empty set
        to True will return None.

    Examples
    --------
    This example will load the example MET data used for unit testing.

    .. code-block:: python

        from act.tests import EXAPLE_MET_YAML, EXAMPLE_MET1
        from act.io.arm import read_arm_netcdf
        from act.qc.add_supplemental_qc import read_yaml_supplemental_qc
        ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=True)
        result = read_yaml_supplemental_qc(ds, EXAPLE_MET_YAML,
                                     variables=['rh_mean'], assessments='Bad')
        print(result)

        {'rh_mean': {'Bad': {'description': 'Values are bad',
            'time': array([['2020-01-01T00:01:02.000', '2020-01-01T00:03:44.000'],
                           ['2020-01-02T00:01:02.000', '2020-01-02T00:03:44.000']],
                           dtype='datetime64[ms]')}}}

    """

    flag_file = []
    if Path(fullpath).is_file():
        flag_file = [Path(fullpath)]
    else:
        try:
            datastream = ds.attrs['_datastream']
        except KeyError:
            raise RuntimeError(
                'Unable to determine datastream name from Dataset. Need to set global attribute '
                '_datastream in Dataset or provided full path to flag file.'
            )

        flag_file = list(Path(fullpath).glob(f'{datastream}.yml'))
        flag_file.extend(list(Path(fullpath).glob(f'{datastream}.yaml')))

    if len(flag_file) > 0:
        flag_file = flag_file[0]
    else:
        if not quiet:
            print(f'Could not find supplemental QC file for {datastream} in {fullpath}')

        return None

    # Ensure keywords are lists
    if isinstance(variables, str):
        variables = [variables]

    if isinstance(assessments, str):
        assessments = [assessments]

    if isinstance(time_delim, str):
        time_delim = [time_delim]

    # Ensure the assessments are capitalized for matching
    if assessments is not None:
        assessments = [ii.capitalize() for ii in assessments]

    # Read YAML file
    with open(flag_file) as fp:
        try:
            data_dict = yaml.load(fp, Loader=yaml.FullLoader)
        except AttributeError:
            data_dict = yaml.load(fp)

    # If variable names are provided only keep those names.
    if variables is not None:
        variables.append('_all')
        del_vars = set(data_dict.keys()) - set(variables)
        for var_name in del_vars:
            del data_dict[var_name]

    # If assessments are given only keep those assessments.
    if assessments is not None:
        for var_name in data_dict.keys():
            for asses_name in data_dict[var_name].keys():
                # Check if yaml file assessments are capitalized. If not fix.
                if not asses_name == asses_name.capitalize():
                    data_dict[var_name][asses_name.capitalize()] = data_dict[var_name][asses_name]
                    del data_dict[var_name][asses_name]

            # Delete assessments if not in provided list.
            del_asses = set(data_dict[var_name].keys()) - set(assessments)
            for asses_name in del_asses:
                del data_dict[var_name][asses_name]

    # Convert from string to numpy datetime64 2D array
    if datetime64:
        for var_name in data_dict.keys():
            for asses_name in data_dict[var_name].keys():
                for description in data_dict[var_name][asses_name].keys():
                    try:
                        num_times = len(data_dict[var_name][asses_name][description])
                        new_times = np.empty((num_times, 2), dtype='datetime64[ms]')
                    except TypeError:
                        # Set to single value Not A Time numpy array if times
                        # from yaml file are empty list.
                        new_times = np.full([], np.datetime64('NaT'), dtype='datetime64[ms]')

                    for ii, tm in enumerate(data_dict[var_name][asses_name][description]):
                        # Split the times on multiple different types of delimiters.
                        for delim in time_delim:
                            split_tm = tm.split(delim)
                            if len(split_tm) > 1:
                                break

                        new_times[ii, 0] = np.datetime64(parser.parse(split_tm[0]))
                        new_times[ii, 1] = np.datetime64(parser.parse(split_tm[1]))

                    data_dict[var_name][asses_name][description] = new_times

    # If the return dictinary is empty convert to None
    if none_if_empty and len(data_dict) == 0:
        data_dict = None

    return data_dict


def apply_supplemental_qc(
    ds,
    fullpath,
    variables=None,
    assessments=None,
    apply_all=True,
    exclude_all_variables=None,
    quiet=False,
):
    """
    Apply flagging from supplemental QC file by adding new QC tests.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset containing data. QC variables should be converted to CF
        format prior to adding new tests.
    fullpath : str or `pathlib.Path`
        Fullpath to file or directory with supplemental QC files.
    variables : str, list of str or None
        Variables to apply to the dataset from supplemental QC flag file.
        If not set will apply all variables in the file.
    assessments : str, list of str or None
        Assessments to apply. If not not set will apply all assesments in the flag file.
    apply_all : boolean
        If a "_all" variable exists in the supplemental QC flag file will apply to all variables
        in the Dataset.
    exclude_all_variables : str, list of str or None
        Variables to skip when applying "_all" variables.
    quiet : boolean
        Suppress information about not finding a supplemental QC file to read.

    Examples
    --------
    This example will load the example sounding data used for unit testing.

    .. code-block:: python

        from act.tests import EXAPLE_MET_YAML, EXAMPLE_MET1
        from act.io.arm import read_arm_netcdf
        from act.qc.add_supplemental_qc import apply_supplemental_qc
        ds = read_arm_netcdf(EXAMPLE_MET1, cleanup_qc=True)
        apply_supplemental_qc(ds, EXAPLE_MET_YAML, apply_all=False)
        print(ds['qc_temp_mean'].attrs['flag_meanings'])

        ['Value is equal to missing_value.', 'Value is less than the fail_min.',
         'Value is greater than the fail_max.',
         'Difference between current and previous values exceeds fail_delta.',
         'Values are bad', 'Values are super bad', 'Values are suspect', 'Values are good']

    """

    exclude_vars = ['time', 'base_time', 'time_offset']
    if exclude_all_variables is not None:
        if isinstance(exclude_all_variables, str):
            exclude_all_variables = [exclude_all_variables]

        exclude_vars.extend(exclude_all_variables)

    flag_dict = read_yaml_supplemental_qc(
        ds, fullpath, variables=variables, assessments=assessments, quiet=quiet
    )

    if flag_dict is None:
        return

    for var_name in list(ds.variables):
        if var_name in flag_dict.keys():
            for asses_name in flag_dict[var_name].keys():
                for description in flag_dict[var_name][asses_name]:
                    times = flag_dict[var_name][asses_name][description]

                    if np.all(np.isnat(times)):
                        continue

                    indexes = np.array([], dtype=np.int32)
                    for vals in times:
                        ind = np.argwhere(
                            (ds['time'].values >= vals[0]) & (ds['time'].values <= vals[1])
                        )

                        if len(ind) > 0:
                            indexes = np.append(indexes, ind)

                    if indexes.size > 0:
                        ds.qcfilter.add_test(
                            var_name,
                            index=indexes,
                            test_meaning=description,
                            test_assessment=asses_name,
                        )

    var_name = '_all'
    if apply_all and var_name in flag_dict.keys():
        for asses_name in flag_dict[var_name].keys():
            for description in flag_dict[var_name][asses_name]:
                times = flag_dict[var_name][asses_name][description]

                if np.all(np.isnat(times)):
                    continue

                indexes = np.array([], dtype=np.int32)
                for vals in times:
                    ind = np.argwhere(
                        (ds['time'].values >= vals[0]) & (ds['time'].values <= vals[1])
                    )
                    if ind.size > 0:
                        indexes = np.append(indexes, np.ndarray.flatten(ind))

                if indexes.size > 0:
                    for all_var_name in list(ds.data_vars):
                        if all_var_name in exclude_vars:
                            continue

                        if 'time' not in ds[all_var_name].dims:
                            continue

                        try:
                            if ds[all_var_name].attrs['standard_name'] == 'quality_flag':
                                continue
                        except KeyError:
                            pass

                        ds.qcfilter.add_test(
                            all_var_name,
                            index=indexes,
                            test_meaning=description,
                            test_assessment=asses_name,
                        )
