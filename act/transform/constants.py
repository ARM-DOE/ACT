"""
QC flag bit constants shared by the ``act.transform`` kernels
(:mod:`act.transform.bin_average`, :mod:`act.transform.interpolate`,
:mod:`act.transform.subsample`).

Each flag is a bit position in a packed integer QC field. Multiple flags
can be set simultaneously via bitwise OR. ``QC_FLAG_MEANINGS`` and
``QC_FLAG_ASSESSMENTS`` and ``QC_FLAG_COMMENTS`` give the ARM-style
``flag_meanings``/``flag_assessments``/``flag_comments`` text for each flag, in
the same order as ``QC_ALL_FLAGS``, so that output QC variables carry the same
metadata as ARM transformation products.

"""

# These values and metadata match ARM's transform QC convention exactly.
QC_BAD = 1
QC_INDETERMINATE = 1 << 1
QC_INTERPOLATE = 1 << 2
QC_EXTRAPOLATE = 1 << 3
QC_NOT_USING_CLOSEST = 1 << 4
QC_SOME_BAD_INPUTS = 1 << 5
QC_ZERO_WEIGHT = 1 << 6
QC_OUTSIDE_RANGE = 1 << 7
QC_ALL_BAD_INPUTS = 1 << 8
QC_BAD_STD = 1 << 9
QC_INDETERMINATE_STD = 1 << 10
QC_BAD_GOODFRAC = 1 << 11
QC_INDETERMINATE_GOODFRAC = 1 << 12

# Ordered (flag_mask, flag_meaning, flag_assessment, flag_comment) values used
# to populate ARM-compatible QC attributes on transform output variables.
QC_ALL_FLAGS = [
    QC_BAD,
    QC_INDETERMINATE,
    QC_INTERPOLATE,
    QC_EXTRAPOLATE,
    QC_NOT_USING_CLOSEST,
    QC_SOME_BAD_INPUTS,
    QC_ZERO_WEIGHT,
    QC_OUTSIDE_RANGE,
    QC_ALL_BAD_INPUTS,
    QC_BAD_STD,
    QC_INDETERMINATE_STD,
    QC_BAD_GOODFRAC,
    QC_INDETERMINATE_GOODFRAC,
]

QC_FLAG_MEANINGS = [
    "QC_BAD:  Transformation could not finish, value set to missing_value.",
    "QC_INDETERMINATE:  Some, or all, of the input values used to create this output value had a QC assessment of Indeterminate.",
    "QC_INTERPOLATE:  Indicates a non-standard interpolation using points other than the two that bracket the target index was applied.",
    "QC_EXTRAPOLATE:  Indicates extrapolation is performed out from two points on the same side of the target index.",
    "QC_NOT_USING_CLOSEST:  Nearest good point is not the nearest actual point.",
    "QC_SOME_BAD_INPUTS:  Some, but not all, of the inputs in the averaging window were flagged as bad and excluded from the transform.",
    "QC_ZERO_WEIGHT:  The weights for all the input points to be averaged for this output bin were set to zero.",
    "QC_OUTSIDE_RANGE:  No input samples exist in the transformation region, value set to missing_value.",
    "QC_ALL_BAD_INPUTS:  All the input values in the transformation region are bad, value set to missing_value.",
    "QC_BAD_STD:  Standard deviation over averaging interval is greater than limit set by transform parameter std_bad_max.",
    "QC_INDETERMINATE_STD:  Standard deviation over averaging interval is greater than limit set by transform parameter std_ind_max.",
    "QC_BAD_GOODFRAC:  Fraction of good and indeterminate points over averaging interval are less than limit set by transform parameter goodfrac_bad_min.",
    "QC_INDETERMINATE_GOODFRAC:  Fraction of good and indeterminate points over averaging interval is less than limit set by transform parameter goodfrac_ind_min.",
]

QC_FLAG_ASSESSMENTS = [
    "Bad",
    "Indeterminate",
    "Indeterminate",
    "Indeterminate",
    "Indeterminate",
    "Indeterminate",
    "Indeterminate",
    "Bad",
    "Bad",
    "Bad",
    "Indeterminate",
    "Bad",
    "Indeterminate",
]

QC_FLAG_COMMENTS = [
    "An example that will trip this bit is if all values are bad or outside range.",
    "",
    "An example of why this may occur is if one or both of the nearest points was flagged as bad.  Applies only to interpolate transformation method.",
    "This occurs because the input grid does not span the output grid, or because all the points within range and on one side of the target were flagged as bad.  Applies only to the interpolate transformation method.",
    "Applies only to subsample transformation method.",
    "Applies only to the bin average transformation method.",
    'The output "average" value is set to zero, independent of the value of the input.  Applies only to bin average transformation method.',
    'Nearest good bracketing points are farther away than the "range" transform parameter if transformation is done using the interpolate or subsample method, or "width" if a bin average transform is applied.  Test can also fail if more than half an input bin is extrapolated beyond the first or last point of the input grid.',
    "The transformation could not be completed. Values in the output grid are set to missing_value and the QC_BAD bit is also set.",
    "Applies only to the bin average transformation method.",
    "Applies only to the bin average transformation method.",
    "Applies only to the bin average transformation method.",
    "Applies only to the bin average transformation method.",
]


def add_arm_qc_attrs(qc_da):
    """Populate ARM-style QC attributes on a transform output QC DataArray.

    Sets ``flag_masks``, ``flag_meanings``, ``flag_assessments``,
    ``flag_comments``, and ``standard_name='quality_flag'`` on ``qc_da`` using
    the metadata convention emitted by ARM transformation products. The
    resulting QC remains compatible with ACT's ``ds.qcfilter``/``ds.clean``
    utilities.

    Parameters
    ----------
    qc_da : xarray.DataArray
        Integer QC DataArray to annotate in place.

    Returns
    -------
    qc_da : xarray.DataArray
        The same DataArray, with QC attributes populated.

    """
    qc_da.attrs["flag_masks"] = list(QC_ALL_FLAGS)
    qc_da.attrs["flag_meanings"] = list(QC_FLAG_MEANINGS)
    qc_da.attrs["flag_assessments"] = list(QC_FLAG_ASSESSMENTS)
    qc_da.attrs["flag_comments"] = list(QC_FLAG_COMMENTS)
    qc_da.attrs["standard_name"] = "quality_flag"
    return qc_da
