"""
Functions and methods for performing solar radiation tests taken from
the BSRN Global Network recommended QC tests, V2.0

https://bsrn.awi.de

"""

import warnings
import numpy as np
import dask.array as da
from scipy.constants import Stefan_Boltzmann

from act.utils.geo_utils import get_solar_azimuth_elevation
from act.utils.data_utils import convert_units


def _calculate_solar_parameters(ds, lat_name, lon_name, solar_constant):
    """
    Function to calculate solar zenith angles and solar constant adjusted
    to Earth Sun distance

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing location variables
    lat_name : str
        Variable name for latitude
    lon_name : str
        Variable name for longitude
    solar_constant : float
        Solar constant in W/m^2

    Returns
    -------
    Tuple containing (solar zenith angle array, solar constant scalar)

    """
    latitude = ds[lat_name].values
    if latitude.size > 1:
        latitude = latitude[0]
    longitude = ds[lon_name].values
    if longitude.size > 1:
        longitude = longitude[0]

    # Calculate solar parameters
    elevation, _, solar_distance = get_solar_azimuth_elevation(
        latitude=latitude, longitude=longitude, time=ds['time'].values
    )
    solar_distance = np.nanmean(solar_distance)
    Sa = solar_constant / solar_distance**2

    sza = 90.0 - elevation

    return (sza, Sa)


def _find_indexes(ds, var_name, min_limit, max_limit, use_dask):
    """
    Function to find array indexes where failing limit tests

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing data to use in test
    var_name : str
        Variable name to inspect
    min_limit : float or numpy array
        Minimum limit to use for returning indexes
    max_limit : float or numpy array
        Maximum limit to use for returning indexes
    use_dask : boolean
        Option to use Dask operations instead of Numpy

    Returns
    -------
    Tuple containing solar zenith angle array and solar constant scalar

    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        if use_dask and isinstance(ds[var_name].data, da.Array):
            index_min = da.where(ds[var_name].data < min_limit, True, False).compute()
            index_max = da.where(ds[var_name].data > max_limit, True, False).compute()
        else:
            index_min = np.less(ds[var_name].values, min_limit)
            index_max = np.greater(ds[var_name].values, max_limit)

    return (index_min, index_max)


class QCTests:
    """
    This is a Mixins class used to allow using qcfilter class that is already
    registered to the Xarray dataset. All the methods in this class will be added
    to the qcfilter class. Doing this to make the code spread across more files
    so it is more manageable and readable. Additinal files of tests can be added
    to qcfilter by creating a new class in the new file and adding to qcfilter
    class declaration.

    """

    def bsrn_limits_test(
        self,
        test='Physically Possible',
        gbl_SW_dn_name=None,
        glb_diffuse_SW_dn_name=None,
        direct_normal_SW_dn_name=None,
        direct_SW_dn_name=None,
        glb_SW_up_name=None,
        glb_LW_dn_name=None,
        glb_LW_up_name=None,
        sw_min_limit=None,
        lw_min_dn_limit=None,
        lw_min_up_limit=None,
        lw_max_dn_limit=None,
        lw_max_up_limit=None,
        solar_constant=1366,
        lat_name='lat',
        lon_name='lon',
        use_dask=False,
    ):
        """
        Method to apply BSRN limits test and add results to ancillary quality control variable.
        Need to provide variable name for each measurement for the test to be performed. If no
        limits provided will use default values. All data must be in W/m^2 units.  Test will
        provided exception if required variable name is missing.

        Parameters
        ----------
        test : str
            Type of tests to apply. Options include "Physically Possible" or "Extremely Rare"
        gbl_SW_dn_name : str
            Variable name in the Dataset for global shortwave downwelling radiation
            measured by unshaded pyranometer
        glb_diffuse_SW_dn_name : str
            Variable name in the Dataset for global diffuse shortwave downwelling radiation
            measured by shaded pyranometer
        direct_normal_SW_dn_name : str
            Variable name in the Dataset for direct normal shortwave downwelling radiation
        direct_SW_dn_name : str
            Variable name in the Dataset for direct shortwave downwelling radiation
        glb_SW_up_name : str
            Variable name in the Dataset for global shortwave upwelling radiation
        glb_LW_dn_name : str
            Variable name in the Dataset for global longwave downwelling radiation
        glb_LW_up_name : str
            Variable name in the Dataset for global longwave upwelling radiation
        sw_min_limit : int or float
            Lower limit for shortwave radiation test
        lw_min_dn_limit : int or float
            Lower limit for downwelling longwave radiation test measured by a pyrgeometer
        lw_min_up_limit : int or float
            Lower limit for upwelling longwave radiation test measured by a pyrgeometer
        lw_max_dn_limit : int or float
            Upper limit for downwelling longwave radiation test measured by a pyrgeometer
        lw_max_up_limit : int or float
            Upper limit for upwelling longwave radiation test measured by a pyrgeometer
        solar_constant : int or float
            Mean solar constant used in upper limit calculation. Earth sun distance will be
            calculated and applied to this value.
        lat_name : str
            Variable name in the Dataset for latitude
        lon_name : str
            Variable name in the Dataset for longitude
        use_dask : boolean
            Option to use Dask for processing if data is stored in a Dask array

        References
        ----------
        Long, Charles N., and Ellsworth G. Dutton. "BSRN Global Network recommended QC tests, V2. x." (2010).

        Examples
        --------
            .. code-block:: python

                ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_BRS, cleanup_qc=True)
                ds.qcfilter.bsrn_limits_test(
                    gbl_SW_dn_name='down_short_hemisp',
                    glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
                    direct_normal_SW_dn_name='short_direct_normal',
                    glb_SW_up_name='up_short_hemisp',
                    glb_LW_dn_name='down_long_hemisp_shaded',
                    glb_LW_up_name='up_long_hemisp')
        """

        test_names_org = ["Physically Possible", "Extremely Rare"]
        test = test.lower()
        test_names = [ii.lower() for ii in test_names_org]
        if test not in test_names:
            raise ValueError(
                f"Value of '{test}' in keyword 'test' not recognized. "
                f"Must a single value in options {test_names_org}"
            )

        sza, Sa = _calculate_solar_parameters(self._ds, lat_name, lon_name, solar_constant)

        if test == test_names[0]:
            if sw_min_limit is None:
                sw_min_limit = -4.0
            if lw_min_dn_limit is None:
                lw_min_dn_limit = 40.0
            if lw_min_up_limit is None:
                lw_min_up_limit = 40.0
            if lw_max_dn_limit is None:
                lw_max_dn_limit = 700.0
            if lw_max_up_limit is None:
                lw_max_up_limit = 900.0
        elif test == test_names[1]:
            if sw_min_limit is None:
                sw_min_limit = -2.0
            if lw_min_dn_limit is None:
                lw_min_dn_limit = 60.0
            if lw_min_up_limit is None:
                lw_min_up_limit = 60.0
            if lw_max_dn_limit is None:
                lw_max_dn_limit = 500.0
            if lw_max_up_limit is None:
                lw_max_up_limit = 700.0

        # Global Shortwave downwelling min and max tests
        if gbl_SW_dn_name is not None:
            cos_sza = np.cos(np.radians(sza))
            cos_sza[sza > 90.0] = 0.0
            if test == test_names[0]:
                sw_max_limit = Sa * 1.5 * cos_sza**1.2 + 100.0
            elif test == test_names[1]:
                sw_max_limit = Sa * 1.2 * cos_sza**1.2 + 50.0

            index_min, index_max = _find_indexes(
                self._ds, gbl_SW_dn_name, sw_min_limit, sw_max_limit, use_dask
            )

            self._ds.qcfilter.add_test(
                gbl_SW_dn_name,
                index=index_min,
                test_assessment='Bad',
                test_meaning=f"Value less than BSRN {test.lower()} limit of {sw_min_limit} W/m^2",
            )

            self._ds.qcfilter.add_test(
                gbl_SW_dn_name,
                index=index_max,
                test_assessment='Bad',
                test_meaning=f"Value greater than BSRN {test.lower()} limit",
            )

        # Diffuse Shortwave downwelling min and max tests
        if glb_diffuse_SW_dn_name is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if test == test_names[0]:
                    sw_max_limit = Sa * 0.95 * np.cos(np.radians(sza)) ** 1.2 + 50.0
                elif test == test_names[1]:
                    sw_max_limit = Sa * 0.75 * np.cos(np.radians(sza)) ** 1.2 + 30.0

            index_min, index_max = _find_indexes(
                self._ds, glb_diffuse_SW_dn_name, sw_min_limit, sw_max_limit, use_dask
            )
            self._ds.qcfilter.add_test(
                glb_diffuse_SW_dn_name,
                index=index_min,
                test_assessment='Bad',
                test_meaning=f"Value less than BSRN {test.lower()} limit of {sw_min_limit} W/m^2",
            )

            self._ds.qcfilter.add_test(
                glb_diffuse_SW_dn_name,
                index=index_max,
                test_assessment='Bad',
                test_meaning=f"Value greater than BSRN {test.lower()} limit",
            )

        # Direct Normal Shortwave downwelling min and max tests
        if direct_normal_SW_dn_name is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if test == test_names[0]:
                    sw_max_limit = Sa
                elif test == test_names[1]:
                    sw_max_limit = Sa * 0.95 * np.cos(np.radians(sza)) ** 0.2 + 10.0

            index_min, index_max = _find_indexes(
                self._ds, direct_normal_SW_dn_name, sw_min_limit, sw_max_limit, use_dask
            )
            self._ds.qcfilter.add_test(
                direct_normal_SW_dn_name,
                index=index_min,
                test_assessment='Bad',
                test_meaning=f"Value less than BSRN {test.lower()} limit of {sw_min_limit} W/m^2",
            )

            self._ds.qcfilter.add_test(
                direct_normal_SW_dn_name,
                index=index_max,
                test_assessment='Bad',
                test_meaning=f"Value greater than BSRN {test.lower()} limit",
            )

        # Direct Shortwave downwelling min and max tests
        if direct_SW_dn_name is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if test == test_names[0]:
                    sw_max_limit = Sa * np.cos(np.radians(sza))
                elif test == test_names[1]:
                    sw_max_limit = Sa * 0.95 * np.cos(np.radians(sza)) ** 1.2 + 10

            index_min, index_max = _find_indexes(
                self._ds, direct_SW_dn_name, sw_min_limit, sw_max_limit, use_dask
            )

            self._ds.qcfilter.add_test(
                direct_SW_dn_name,
                index=index_min,
                test_assessment='Bad',
                test_meaning=f"Value less than BSRN {test.lower()} limit of {sw_min_limit} W/m^2",
            )

            self._ds.qcfilter.add_test(
                direct_SW_dn_name,
                index=index_max,
                test_assessment='Bad',
                test_meaning=f"Value greater than BSRN {test.lower()} limit",
            )

        # Shortwave up welling min and max tests
        if glb_SW_up_name is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if test == test_names[0]:
                    sw_max_limit = Sa * 1.2 * np.cos(np.radians(sza)) ** 1.2 + 50
                elif test == test_names[1]:
                    sw_max_limit = Sa * np.cos(np.radians(sza)) ** 1.2 + 50

            index_min, index_max = _find_indexes(
                self._ds, glb_SW_up_name, sw_min_limit, sw_max_limit, use_dask
            )

            self._ds.qcfilter.add_test(
                glb_SW_up_name,
                index=index_min,
                test_assessment='Bad',
                test_meaning=f"Value less than BSRN {test.lower()} limit of {sw_min_limit} W/m^2",
            )

            self._ds.qcfilter.add_test(
                glb_SW_up_name,
                index=index_max,
                test_assessment='Bad',
                test_meaning=f"Value greater than BSRN {test.lower()} limit",
            )

        # Longwave downwelling min and max tests
        if glb_LW_dn_name is not None:
            index_min, index_max = _find_indexes(
                self._ds, glb_LW_dn_name, lw_min_dn_limit, lw_max_dn_limit, use_dask
            )

            self._ds.qcfilter.add_test(
                glb_LW_dn_name,
                index=index_min,
                test_assessment='Bad',
                test_meaning=f"Value less than BSRN {test.lower()} limit of {lw_min_dn_limit} W/m^2",
            )

            self._ds.qcfilter.add_test(
                glb_LW_dn_name,
                index=index_max,
                test_assessment='Bad',
                test_meaning=f"Value greater than BSRN {test.lower()} limit of {lw_max_dn_limit} W/m^2",
            )

        # Longwave upwelling min and max tests
        if glb_LW_up_name is not None:
            index_min, index_max = _find_indexes(
                self._ds, glb_LW_up_name, lw_min_up_limit, lw_max_up_limit, use_dask
            )

            self._ds.qcfilter.add_test(
                glb_LW_up_name,
                index=index_min,
                test_assessment='Bad',
                test_meaning=f"Value less than BSRN {test.lower()} limit of {lw_min_up_limit} W/m^2",
            )

            self._ds.qcfilter.add_test(
                glb_LW_up_name,
                index=index_max,
                test_assessment='Bad',
                test_meaning=f"Value greater than BSRN {test.lower()} limit of {lw_max_up_limit} W/m^2",
            )

    def bsrn_comparison_tests(
        self,
        test,
        gbl_SW_dn_name=None,
        glb_diffuse_SW_dn_name=None,
        direct_normal_SW_dn_name=None,
        glb_SW_up_name=None,
        glb_LW_dn_name=None,
        glb_LW_up_name=None,
        air_temp_name=None,
        test_assessment='Indeterminate',
        lat_name='lat',
        lon_name='lon',
        LWdn_lt_LWup_component=25.0,
        LWdn_gt_LWup_component=300.0,
        use_dask=False,
    ):
        """
        Method to apply BSRN comparison tests and add results to ancillary quality control variable.
        Need to provided variable name for each measurement for the test to be performed. All radiation
        data must be in W/m^2 units. Test will provided exception if required variable name is missing.

        Parameters
        ----------
        test : str
            Type of tests to apply. Options include: 'Global over Sum SW Ratio', 'Diffuse Ratio',
            'SW up', 'LW down to air temp', 'LW up to air temp', 'LW down to LW up'
        gbl_SW_dn_name : str
            Variable name in Dataset for global shortwave downwelling radiation
            measured by unshaded pyranometer
        glb_diffuse_SW_dn_name : str
            Variable name in Dataset for global diffuse shortwave downwelling radiation
            measured by shaded pyranometer
        direct_normal_SW_dn_name : str
            Variable name in Dataset for direct normal shortwave downwelling radiation
        glb_SW_up_name : str
            Variable name in Dataset for global shortwave upwelling radiation
        glb_LW_dn_name : str
            Variable name in Dataset for global longwave downwelling radiation
        glb_LW_up_name : str
            Variable name in Dataset for global longwave upwelling radiation
        air_temp_name : str
            Variable name in Dataset for atmospheric air temperature. Variable used
            in longwave tests.
        test_assessment : str
            Test assessment string value appended to flag_assessments attribute of QC variable.
        lat_name : str
            Variable name in the Dataset for latitude
        lon_name : str
            Variable name in the Dataset for longitude
        LWdn_lt_LWup_component : int or float
            Value used in longwave down less than longwave up test.
        LWdn_gt_LWup_component : int or float
            Value used in longwave down greater than longwave up test.
        use_dask : boolean
            Option to use Dask for processing if data is stored in a Dask array

        References
        ----------
        Long, Charles N., and Ellsworth G. Dutton. "BSRN Global Network recommended QC tests, V2. x." (2010).

        Examples
        --------
            .. code-block:: python

                ds = act.io.arm.read_arm_netcdf(act.tests.EXAMPLE_BRS, cleanup_qc=True)
                ds.qcfilter.bsrn_comparison_tests(
                    gbl_SW_dn_name='down_short_hemisp',
                    glb_diffuse_SW_dn_name='down_short_diffuse_hemisp',
                    direct_normal_SW_dn_name='short_direct_normal',
                    glb_SW_up_name='up_short_hemisp',
                    glb_LW_dn_name='down_long_hemisp_shaded',
                    glb_LW_up_name='up_long_hemisp',
                    use_dask=True)
        """

        if isinstance(test, str):
            test = [test]

        test_options = [
            'Global over Sum SW Ratio',
            'Diffuse Ratio',
            'SW up',
            'LW down to air temp',
            'LW up to air temp',
            'LW down to LW up',
        ]

        solar_constant = 1360.8
        sza, Sa = _calculate_solar_parameters(self._ds, lat_name, lon_name, solar_constant)

        # Ratio of Global over Sum SW
        if test_options[0] in test:
            if (
                gbl_SW_dn_name is None
                or glb_diffuse_SW_dn_name is None
                or direct_normal_SW_dn_name is None
            ):
                raise ValueError(
                    'Must set keywords gbl_SW_dn_name, glb_diffuse_SW_dn_name, '
                    f'direct_normal_SW_dn_name for {test_options[0]} test.'
                )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if use_dask and isinstance(self._ds[glb_diffuse_SW_dn_name].data, da.Array):
                    sum_sw_down = self._ds[glb_diffuse_SW_dn_name].data + self._ds[
                        direct_normal_SW_dn_name
                    ].data * np.cos(np.radians(sza))
                    sum_sw_down[sum_sw_down < 50] = np.nan
                    ratio = self._ds[gbl_SW_dn_name].data / sum_sw_down
                    index_a = sza < 75
                    index_1 = da.where((ratio > 1.08) & index_a, True, False)
                    index_2 = da.where((ratio < 0.92) & index_a, True, False)
                    index_b = (sza >= 75) & (sza < 93)
                    index_3 = da.where((ratio > 1.15) & index_b & index_b, True, False)
                    index_4 = da.where((ratio < 0.85) & index_b, True, False)
                    index = (index_1 | index_2 | index_3 | index_4).compute()
                else:
                    sum_sw_down = self._ds[glb_diffuse_SW_dn_name].values + self._ds[
                        direct_normal_SW_dn_name
                    ].values * np.cos(np.radians(sza))
                    sum_sw_down[sum_sw_down < 50] = np.nan
                    ratio = self._ds[gbl_SW_dn_name].values / sum_sw_down
                    index_a = sza < 75
                    index_1 = (ratio > 1.08) & index_a
                    index_2 = (ratio < 0.92) & index_a
                    index_b = (sza >= 75) & (sza < 93)
                    index_3 = (ratio > 1.15) & index_b
                    index_4 = (ratio < 0.85) & index_b
                    index = index_1 | index_2 | index_3 | index_4

            test_meaning = "Ratio of Global over Sum shortwave larger than expected"
            self._ds.qcfilter.add_test(
                gbl_SW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )
            self._ds.qcfilter.add_test(
                glb_diffuse_SW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )
            self._ds.qcfilter.add_test(
                direct_normal_SW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )

        # Diffuse Ratio
        if test_options[1] in test:
            if gbl_SW_dn_name is None or glb_diffuse_SW_dn_name is None:
                raise ValueError(
                    'Must set keywords gbl_SW_dn_name, glb_diffuse_SW_dn_name '
                    f'for {test_options[1]} test.'
                )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if use_dask and isinstance(self._ds[glb_diffuse_SW_dn_name].data, da.Array):
                    ratio = self._ds[glb_diffuse_SW_dn_name].data / self._ds[gbl_SW_dn_name].data
                    ratio[self._ds[gbl_SW_dn_name].data < 50] = np.nan
                    index_a = sza < 75
                    index_1 = da.where((ratio >= 1.05) & index_a, True, False)
                    index_b = (sza >= 75) & (sza < 93)
                    index_2 = da.where((ratio >= 1.10) & index_b, True, False)
                    index = (index_1 | index_2).compute()
                else:
                    ratio = (
                        self._ds[glb_diffuse_SW_dn_name].values / self._ds[gbl_SW_dn_name].values
                    )
                    ratio[self._ds[gbl_SW_dn_name].values < 50] = np.nan
                    index_a = sza < 75
                    index_1 = (ratio >= 1.05) & index_a
                    index_b = (sza >= 75) & (sza < 93)
                    index_2 = (ratio >= 1.10) & index_b
                    index = index_1 | index_2

            test_meaning = "Ratio of Diffuse Shortwave over Global Shortwave larger than expected"
            self._ds.qcfilter.add_test(
                gbl_SW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )
            self._ds.qcfilter.add_test(
                glb_diffuse_SW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )

        # Shortwave up comparison
        if test_options[2] in test:
            if (
                glb_SW_up_name is None
                or glb_diffuse_SW_dn_name is None
                or direct_normal_SW_dn_name is None
            ):
                raise ValueError(
                    'Must set keywords glb_SW_up_name, glb_diffuse_SW_dn_name, '
                    f'direct_normal_SW_dn_name for {test_options[2]} test.'
                )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if use_dask and isinstance(self._ds[glb_diffuse_SW_dn_name].data, da.Array):
                    sum_sw_down = self._ds[glb_diffuse_SW_dn_name].data + self._ds[
                        direct_normal_SW_dn_name
                    ].data * np.cos(np.radians(sza))

                    sum_sw_down[sum_sw_down < 50] = np.nan
                    index = da.where(
                        self._ds[glb_SW_up_name].data > sum_sw_down, True, False
                    ).compute()
                else:
                    sum_sw_down = self._ds[glb_diffuse_SW_dn_name].values + self._ds[
                        direct_normal_SW_dn_name
                    ].values * np.cos(np.radians(sza))
                    sum_sw_down[sum_sw_down < 50] = np.nan
                    index = self._ds[glb_SW_up_name].values > sum_sw_down

            test_meaning = "Ratio of Shortwave Upwelling greater than Shortwave Sum"
            self._ds.qcfilter.add_test(
                glb_SW_up_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )
            self._ds.qcfilter.add_test(
                glb_diffuse_SW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )
            self._ds.qcfilter.add_test(
                direct_normal_SW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )

        # Longwave down to air temperature comparison
        if test_options[3] in test:
            if glb_LW_dn_name is None or air_temp_name is None:
                raise ValueError(
                    'Must set keywords glb_LW_dn_name, air_temp_name '
                    f' for {test_options[3]} test.'
                )

            air_temp = convert_units(
                self._ds[air_temp_name].values, self._ds[air_temp_name].attrs['units'], 'degK'
            )
            if use_dask and isinstance(self._ds[glb_LW_dn_name].data, da.Array):
                air_temp = da.array(air_temp)
                conversion = da.array(Stefan_Boltzmann * air_temp**4)
                index_1 = (0.4 * conversion) > self._ds[glb_LW_dn_name].data
                index_2 = (conversion + 25.0) < self._ds[glb_LW_dn_name].data
                index = (index_1 | index_2).compute()
            else:
                conversion = Stefan_Boltzmann * air_temp**4
                index_1 = (0.4 * conversion) > self._ds[glb_LW_dn_name].values
                index_2 = (conversion + 25.0) < self._ds[glb_LW_dn_name].values
                index = index_1 | index_2

            test_meaning = (
                "Longwave downwelling comparison to air temperature out side of expected range"
            )
            self._ds.qcfilter.add_test(
                glb_LW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )

        # Longwave up to air temperature comparison
        if test_options[4] in test:
            if glb_LW_up_name is None or air_temp_name is None:
                raise ValueError(
                    'Must set keywords glb_LW_up_name, air_temp_name '
                    f'for {test_options[3]} test.'
                )

            air_temp = convert_units(
                self._ds[air_temp_name].values, self._ds[air_temp_name].attrs['units'], 'degK'
            )
            if use_dask and isinstance(self._ds[glb_LW_up_name].data, da.Array):
                air_temp = da.array(air_temp)
                index_1 = (Stefan_Boltzmann * (air_temp - 15) ** 4) > self._ds[glb_LW_up_name].data
                index_2 = (Stefan_Boltzmann * (air_temp + 25) ** 4) < self._ds[glb_LW_up_name].data
                index = (index_1 | index_2).compute()
            else:
                index_1 = (Stefan_Boltzmann * (air_temp - 15) ** 4) > self._ds[
                    glb_LW_up_name
                ].values
                index_2 = (Stefan_Boltzmann * (air_temp + 25) ** 4) < self._ds[
                    glb_LW_up_name
                ].values
                index = index_1 | index_2

            test_meaning = (
                "Longwave upwelling comparison to air temperature out side of expected range"
            )
            self._ds.qcfilter.add_test(
                glb_LW_up_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )

        # Lonwave down to longwave up comparison
        if test_options[5] in test:
            if glb_LW_dn_name is None or glb_LW_up_name is None:
                raise ValueError(
                    'Must set keywords glb_LW_dn_name, glb_LW_up_name '
                    f'for {test_options[3]} test.'
                )

            if use_dask and isinstance(self._ds[glb_LW_dn_name].data, da.Array):
                index_1 = da.where(
                    self._ds[glb_LW_dn_name].data
                    > (self._ds[glb_LW_up_name].data + LWdn_lt_LWup_component),
                    True,
                    False,
                )
                index_2 = da.where(
                    self._ds[glb_LW_dn_name].data
                    < (self._ds[glb_LW_up_name].data - LWdn_gt_LWup_component),
                    True,
                    False,
                )
                index = (index_1 | index_2).compute()
            else:
                index_1 = self._ds[glb_LW_dn_name].values > (
                    self._ds[glb_LW_up_name].values + LWdn_lt_LWup_component
                )
                index_2 = self._ds[glb_LW_dn_name].values < (
                    self._ds[glb_LW_up_name].values - LWdn_gt_LWup_component
                )
                index = index_1 | index_2

            test_meaning = (
                "Lonwave downwelling compared to longwave upwelling outside of expected range"
            )
            self._ds.qcfilter.add_test(
                glb_LW_dn_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )
            self._ds.qcfilter.add_test(
                glb_LW_up_name,
                index=index,
                test_assessment=test_assessment,
                test_meaning=test_meaning,
            )
