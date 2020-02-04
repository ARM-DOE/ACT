""" Retrieval functions for AERI. """

import numpy as np
from scipy.optimize import brentq
from act.retrievals.radiance_temperature_conversion import planck_converter


def aeri2irt_response_function():
    """
    Function to return the wavenumber and response function to use with AERI
    to IRT conversion.

    Returns
    -------
    tuple : (wavenumber, response function)
        The wavenumber and response function values as numpy float arrays.

    """
    # Fill response function values.  First wavenumbers then response fraction. --;
    wnum = np.array(
        [847.133, 847.615, 848.097, 848.579, 849.061, 849.543, 850.026, 850.508,
         850.990, 851.472, 851.954, 852.436, 852.918, 853.401, 853.883, 854.365, 854.847,
         855.329, 855.811, 856.293, 856.776, 857.258, 857.740, 858.222, 858.704, 859.186,
         859.668, 860.151, 860.633, 861.115, 861.597, 862.079, 862.561, 863.043, 863.526,
         864.008, 864.490, 864.972, 865.454, 865.936, 866.419, 866.901, 867.383, 867.865,
         868.347, 868.829, 869.311, 869.794, 870.276, 870.758, 871.240, 871.722, 872.204,
         872.686, 873.169, 873.651, 874.133, 874.615, 875.097, 875.579, 876.061, 876.544,
         877.026, 877.508, 877.990, 878.472, 878.954, 879.437, 879.919, 880.401, 880.883,
         881.365, 881.847, 882.329, 882.812, 883.294, 883.776, 884.258, 884.740, 885.222,
         885.704, 886.187, 886.669, 887.151, 887.633, 888.115, 888.597, 889.079, 889.562,
         890.044, 890.526, 891.008, 891.490, 891.972, 892.454, 892.937, 893.419, 893.901,
         894.383, 894.865, 895.347, 895.829, 896.312, 896.794, 897.276, 897.758, 898.240,
         898.722, 899.205, 899.687, 900.169, 900.651, 901.133, 901.615, 902.097, 902.580,
         903.062, 903.544, 904.026, 904.508, 904.990, 905.472, 905.955, 906.437, 906.919,
         907.401, 907.883, 908.365, 908.847, 909.330, 909.812, 910.294, 910.776, 911.258,
         911.740, 912.223, 912.705, 913.187, 913.669, 914.151, 914.633, 915.115, 915.598,
         916.080, 916.562, 917.044, 917.526, 918.008, 918.490, 918.973, 919.455, 919.937,
         920.419, 920.901, 921.383, 921.865, 922.348, 922.830, 923.312, 923.794, 924.276,
         924.758, 925.240, 925.723, 926.205, 926.687, 927.169, 927.651, 928.133, 928.615,
         929.098, 929.580, 930.062, 930.544, 931.026, 931.508, 931.991, 932.473, 932.955,
         933.437, 933.919, 934.401, 934.883, 935.366, 935.848, 936.330, 936.812, 937.294,
         937.776, 938.258, 938.741, 939.223, 939.705, 940.187, 940.669, 941.151, 941.633,
         942.116, 942.598, 943.080, 943.562, 944.044, 944.526, 945.009, 945.491, 945.973,
         946.455, 946.937, 947.419, 947.901, 948.384, 948.866, 949.348, 949.830, 950.312,
         950.794, 951.276, 951.759, 952.241, 952.723, 953.205, 953.687, 954.169, 954.651,
         955.134, 955.616, 956.098, 956.580, 957.062, 957.544, 958.026, 958.509, 958.991,
         959.473, 959.955, 960.437, 960.919, 961.401, 961.884, 962.366, 962.848, 963.330,
         963.812, 964.294, 964.777, 965.259, 965.741, 966.223, 966.705, 967.187, 967.669,
         968.152, 968.634, 969.116, 969.598, 970.080, 970.562, 971.044, 971.527, 972.009,
         972.491, 972.973, 973.455, 973.937, 974.419, 974.902, 975.384, 975.866, 976.348,
         976.830, 977.312, 977.795, 978.277, 978.759, 979.241, 979.723, 980.205, 980.687,
         981.170, 981.652, 982.134, 982.616, 983.098, 983.580, 984.062, 984.545, 985.027,
         985.509, 985.991, 986.473, 986.955, 987.438, 987.920, 988.402, 988.884, 989.366,
         989.848, 990.330, 990.812, 991.295, 991.777, 992.259, 992.741, 993.223, 993.705,
         994.188, 994.670, 995.152, 995.634, 996.116, 996.598, 997.080, 997.563, 998.045,
         998.527, 999.009, 999.491, 999.973, 1000.455, 1000.938, 1001.420, 1001.902, 1002.384,
         1002.866, 1003.348, 1003.830, 1004.313, 1004.795, 1005.277, 1005.759, 1006.241,
         1006.720, 1007.210, 1007.690, 1008.170, 1008.650, 1009.130, 1009.620, 1010.100,
         1010.580, 1011.060, 1011.540, 1012.030, 1012.510, 1012.990, 1013.470, 1013.960,
         1014.440, 1014.920, 1015.400, 1015.880, 1016.370, 1016.850, 1017.330, 1017.810,
         1018.290, 1018.780, 1019.260, 1019.740, 1020.220, 1020.710, 1021.190, 1021.670,
         1022.150, 1022.630, 1023.120, 1023.600, 1024.080, 1024.560, 1025.040, 1025.530,
         1026.010, 1026.490, 1026.970, 1027.460, 1027.940, 1028.420, 1028.900, 1029.380,
         1029.870, 1030.350, 1030.830, 1031.310, 1031.800, 1032.280, 1032.760, 1033.240,
         1033.720, 1034.210, 1034.690, 1035.170, 1035.650, 1036.130, 1036.620, 1037.100,
         1037.580, 1038.060, 1038.550, 1039.030, 1039.510, 1039.990, 1040.470, 1040.960,
         1041.440, 1041.920, 1042.400, 1042.880, 1043.370, 1043.850, 1044.330, 1044.810,
         1045.300, 1045.780, 1046.260, 1046.740, 1047.220, 1047.710, 1048.190, 1048.670,
         1049.150, 1049.630, 1050.120, 1050.600, 1051.080, 1051.560, 1052.050, 1052.530,
         1053.010, 1053.490, 1053.970, 1054.460, 1054.940, 1055.420, 1055.900, 1056.380,
         1056.870, 1057.350, 1057.830, 1058.310, 1058.800, 1059.280, 1059.760, 1060.240,
         1060.720, 1061.210, 1061.690, 1062.170, 1062.650, 1063.130, 1063.620, 1064.100],
        dtype=np.float32)

    rf = np.array(
        [0.00000000, 0.00174289, 0.00710459, 0.01246630, 0.01782730, 0.02318830,
         0.02855000, 0.03391170, 0.03927280, 0.04463380, 0.04996460, 0.05523600, 0.06050680,
         0.06577760, 0.07104910, 0.07632050, 0.08159130, 0.08686210, 0.09213360, 0.09740500,
         0.10267600, 0.10794700, 0.11321800, 0.11849000, 0.12376000, 0.12903100, 0.13430300,
         0.13957400, 0.14582200, 0.16096000, 0.17610100, 0.19124100, 0.20638000, 0.22151900,
         0.23665900, 0.25180000, 0.26693800, 0.28207700, 0.29721700, 0.31235800, 0.32749600,
         0.34263500, 0.35777600, 0.37291600, 0.38805500, 0.40319300, 0.41833400, 0.43254500,
         0.44572400, 0.45890300, 0.47208400, 0.48526500, 0.49844400, 0.51162300, 0.52480400,
         0.53798400, 0.55116300, 0.56434300, 0.57752300, 0.59070400, 0.60388300, 0.61706200,
         0.63024300, 0.64342400, 0.65660300, 0.66978200, 0.67679500, 0.67641800, 0.67604100,
         0.67566500, 0.67528800, 0.67491100, 0.67453400, 0.67415800, 0.67378100, 0.67340400,
         0.67302800, 0.67265100, 0.67227400, 0.67189800, 0.67152100, 0.67114400, 0.67076700,
         0.67039100, 0.67001400, 0.66962700, 0.66894000, 0.66825200, 0.66756500, 0.66687800,
         0.66619000, 0.66550300, 0.66481600, 0.66412900, 0.66344100, 0.66275400, 0.66206700,
         0.66137900, 0.66069200, 0.66000500, 0.65931700, 0.65863000, 0.65794300, 0.65725500,
         0.65656800, 0.65601000, 0.65599700, 0.65598400, 0.65597200, 0.65595900, 0.65594700,
         0.65593400, 0.65592200, 0.65590900, 0.65589600, 0.65588400, 0.65587100, 0.65585900,
         0.65584600, 0.65583400, 0.65582100, 0.65580800, 0.65579600, 0.65578300, 0.65577100,
         0.65575800, 0.65619900, 0.65668200, 0.65716400, 0.65764600, 0.65812900, 0.65861100,
         0.65909300, 0.65957600, 0.66005800, 0.66054000, 0.66102300, 0.66150500, 0.66198700,
         0.66247000, 0.66295200, 0.66343400, 0.66391700, 0.66439900, 0.66488100, 0.66536400,
         0.66581700, 0.66615300, 0.66648900, 0.66682400, 0.66716000, 0.66749600, 0.66783200,
         0.66816700, 0.66850300, 0.66883900, 0.66917500, 0.66951000, 0.66984600, 0.67018200,
         0.67051800, 0.67085300, 0.67118900, 0.67152500, 0.67186100, 0.67219700, 0.67253200,
         0.67285600, 0.67236500, 0.67187500, 0.67138400, 0.67089400, 0.67040300, 0.66991300,
         0.66942200, 0.66893200, 0.66844100, 0.66795100, 0.66746000, 0.66697000, 0.66647900,
         0.66598900, 0.66549800, 0.66500800, 0.66451700, 0.66402700, 0.66353600, 0.66304600,
         0.66255500, 0.66200900, 0.66136400, 0.66071900, 0.66007400, 0.65942800, 0.65878300,
         0.65813800, 0.65749300, 0.65684700, 0.65620200, 0.65555700, 0.65491200, 0.65426600,
         0.65362100, 0.65297600, 0.65233100, 0.65168500, 0.65104000, 0.65039500, 0.64975000,
         0.64910400, 0.64845900, 0.64779500, 0.64705600, 0.64631700, 0.64557900, 0.64484000,
         0.64410100, 0.64336200, 0.64262400, 0.64188500, 0.64114600, 0.64040700, 0.63966900,
         0.63893000, 0.63819100, 0.63745200, 0.63671400, 0.63597500, 0.63523600, 0.63449700,
         0.63375800, 0.63302000, 0.63228100, 0.63154200, 0.63074100, 0.62988800, 0.62903400,
         0.62818100, 0.62732700, 0.62647400, 0.62562000, 0.62476700, 0.62391300, 0.62306000,
         0.62220600, 0.62135300, 0.62049900, 0.61964600, 0.61879300, 0.61793900, 0.61708600,
         0.61623200, 0.61537900, 0.61452500, 0.61367200, 0.61281800, 0.61196500, 0.61118100,
         0.61052400, 0.60986700, 0.60921100, 0.60855400, 0.60789700, 0.60724100, 0.60658400,
         0.60592700, 0.60527100, 0.60461400, 0.60395700, 0.60330100, 0.60264400, 0.60198700,
         0.60133000, 0.60067400, 0.60001700, 0.59936000, 0.59870400, 0.59804700, 0.59739000,
         0.59673400, 0.59607700, 0.59582900, 0.59583300, 0.59583700, 0.59584100, 0.59584500,
         0.59585000, 0.59585400, 0.59585800, 0.59586200, 0.59586600, 0.59587100, 0.59587500,
         0.59587900, 0.59588300, 0.59588700, 0.59589200, 0.59589600, 0.59590000, 0.59590400,
         0.59590800, 0.59591300, 0.59591700, 0.59592100, 0.59592500, 0.59547600, 0.59403900,
         0.59260100, 0.59116400, 0.58972700, 0.58828900, 0.58685200, 0.58541400, 0.58397700,
         0.58254000, 0.58110200, 0.57966500, 0.57822800, 0.57679000, 0.57535300, 0.57391500,
         0.57247800, 0.57104000, 0.56960300, 0.56816600, 0.56672800, 0.56529100, 0.56385300,
         0.56241600, 0.56097900, 0.55884500, 0.55577200, 0.55270000, 0.54962700, 0.54655400,
         0.54348100, 0.54040900, 0.53733600, 0.53426300, 0.53119000, 0.52811700, 0.52504500,
         0.52197200, 0.51889900, 0.51582600, 0.51275400, 0.50968100, 0.50660800, 0.50353500,
         0.50046300, 0.49739000, 0.49431700, 0.49124400, 0.48817200, 0.48509900, 0.48202600,
         0.47522600, 0.46813400, 0.46104100, 0.45394800, 0.44685600, 0.43976400, 0.43267100,
         0.42557800, 0.41848600, 0.41139400, 0.40430100, 0.39720800, 0.39011600, 0.38302300,
         0.37593000, 0.36883700, 0.36174400, 0.35465300, 0.34756000, 0.34046700, 0.33337600,
         0.32628300, 0.31919000, 0.31209700, 0.30500400, 0.29791300, 0.29130000, 0.28481200,
         0.27832500, 0.27183700, 0.26534900, 0.25886000, 0.25237200, 0.24588600, 0.23939700,
         0.23290900, 0.22642200, 0.21993400, 0.21344600, 0.20695800, 0.20047000, 0.19398300,
         0.18749500, 0.18100700, 0.17452000, 0.16803200, 0.16154300, 0.15505500, 0.14856700,
         0.14208000, 0.13559200, 0.12910400, 0.12268400, 0.11988200, 0.11708000, 0.11427900,
         0.11147700, 0.10867600, 0.10587400, 0.10307200, 0.10027100, 0.09746960, 0.09466790,
         0.09186620, 0.08906450, 0.08626350, 0.08346180, 0.08066010, 0.07785910, 0.07505740,
         0.07225560, 0.06945390, 0.06665220, 0.06385120, 0.06104950, 0.05824780, 0.05544680,
         0.05264510, 0.04984330, 0.04704160, 0.04397770, 0.04070670, 0.03743490, 0.03416300,
         0.03089200, 0.02762020, 0.02434830, 0.02107650, 0.01780460, 0.01453360, 0.01126180,
         0.00798992, 0.00471891, 0.00144707, 0.00000000],
        dtype=np.float32)

    return wnum, rf


def sum_function_aeri(temperature, inTotal):
    """
    Function used with brentq to invert radiance values into temperatures.

    Parameters
    ----------
    temperature : float
        Temperature value to convert to radiance value for comparison.
    inTotal : float
        Radiance value to compare to converted temperature value to find
        inverted minimum.

    Returns
    -------
    float
        Difference between temperatrue value converte to radiance and
        the radiance value.

    """
    rf_wnum, rf = aeri2irt_response_function()
    rad = planck_converter(rf_wnum, temperature=temperature) * rf
    return np.nansum(rad) - inTotal


def aeri2irt(aeri_ds, wnum_name='wnum', rad_name='mean_rad', hatch_name='hatchOpen',
             tolerance=0.1, temp_low=150.0, temp_high=320.0, maxiter=200):
    """
    This function will integrate over the correct wavenumber values to produce
    the effective IRT temperature.

    As a note from the ARM IRT Instrument Handbook
    A positive bias of the sky temperature is exhibited by the downwelling IRT,
    compared to the AERI, during clear-sky conditions when the sky temperature
    is less than ~180K. The effect depends on the characteristics of the
    individual IRT and the internal reference temperature of the IRT. The
    greatest difference compared to AERI will occur when the sky is very clear,
    dry, and cold and the ambient temperature is relatively hot, maximizing the
    difference in temperature between the sky and instrument, and the
    calibration of the IRT at the lower limit of 223K was not performed
    accurately. This bias is especially apparent at high-latitude sites
    (e.g., NSA, OLI, and AWR).

    https://www.arm.gov/publications/tech_reports/handbooks/irt_handbook.pdf

    Author - Ken Kehoe

    Parameters
    ----------
    aeri_ds : Xarray Dataset Object
        The Dataset object containing AERI data.
    wnum_name : str
        The variable name for coordinate dimention of wave number Xarray Dataset.
    hatch_name : str or None
        The variable name for hatch status. If set to None will not try to set
        when hatch is not opent to NaN.
    rad_name : str
        The variable name for mean radiance in Xarray Dataset.
    tolerance : float
        The tolerance value to try and match for returned temperature.
    temp_low : float
        The initial low value to use in zbren function to invert radiances.
    temp_high : float
        The initial low value to use in zbren function to invert radiances.
    maxiter : int
        The maximum number if iterations to use with invertion process.
        Prevents runaway processes.

    Returns
    -------
    obj : Xarray Dataset Object or None
        The aeri_ds Dataset with new DataArray of temperatures added under
        variable name 'aeri_irt_equiv_temperature'.

    """
    # Get data values
    rf_wnum, rf = aeri2irt_response_function()
    wnum = aeri_ds[wnum_name].values
    mean_rad = aeri_ds[rad_name].values

    # Pull out AERI data for correct wavenumbers and apply response function --;
    index = np.where((wnum >= (rf_wnum[0] - 0.001)) & (wnum <= (rf_wnum[-1] + 0.001)))[0]
    if index.size == 0:
        print(f"\n -- Unable to match any AERI wave numbers in aeri2irt for conversion to IRT.\n")
        return None

    wnum = wnum[index]
    mean_rad = mean_rad[:, index]
    # If the wavenumbers in AERI data are not close enough to response function
    # match the wavenumbers and adjust.
    atol = 0.001
    if not np.all(np.isclose(wnum, rf_wnum, atol=atol)):
        index_wnum = []
        index_rf = []
        for ii in range(wnum.size):
            idx = (np.abs(wnum[ii] - rf_wnum)).argmin()
            if np.isclose(wnum[ii], rf_wnum[idx], atol=atol):
                index_wnum.append(ii)
                index_rf.append(idx)
        mean_rad = mean_rad[:, index_wnum]
        rf = rf[index_rf]

    # Apply response function to AERI radiance
    mean_rad = mean_rad * rf

    # Sum along wavenumber dimention to get a single value for each time step
    mean_rad = np.nansum(mean_rad, axis=1)

    # Loop over each time step of the AERI data and through the use of
    # solving for zero determine the AERI equivlante IRT sky temperature.
    aeri_irt_vals = np.full(mean_rad.size, np.nan, dtype=mean_rad.dtype)

    # Look for when the hatch is not in Open position and set values to NaN.
    if hatch_name is not None:
        flag_values = aeri_ds[hatch_name].attrs['flag_values']
        flag_meanings = aeri_ds[hatch_name].attrs['flag_meanings']
        if not isinstance(flag_meanings, list):
            flag_meanings = flag_meanings.split()
        flag_meanings = [att.lower() for att in flag_meanings]
        if not isinstance(flag_values, list):
            flag_values = flag_values.split()
            flag_values = [int(att) for att in flag_values]
        value = flag_values[flag_meanings.index('open')]
        mean_rad[aeri_ds[hatch_name].values != value] = np.nan

    for ii in range(mean_rad.size):
        if np.isnan(mean_rad[ii]):
            continue
        else:
            try:
                aeri_irt_vals[ii] = brentq(sum_function_aeri, temp_low, temp_high,
                                           args=(mean_rad[ii], ), xtol=tolerance, maxiter=maxiter)
            except ValueError:
                pass

    # Add new values to Xarray Dataset
    aeri_ds['aeri_irt_equiv_temperature'] = (
        'time', aeri_irt_vals,
        {'long_name': 'Derived IRT equivalent temperatrues from AERI',
         'units': 'K'})

    return aeri_ds
