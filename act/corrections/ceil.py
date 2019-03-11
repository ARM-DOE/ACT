import numpy as np


def correct_ceil(arm_obj, fill_value=1e-7):
    """
    This procedure corrects celiometer data by filling all zero and
    negative values of backscatter with fill_value and then converting
    the backscatter data into logarithmic space.

    Parameters
    ----------
    arm_obj: ARM Dataset object
        The ceiliometer dataset to correct. The backscatter data should be
        in linear space.
    fill_value: float
        The fill_value to use. The fill_value is entered in linear space.

    Returns
    -------
    arm_obj: ARM Dataset object
        The celiometer dataset containing the corrected values.
    """

    backscat = arm_obj['backscatter'].data
    backscat[backscat <= 0] = fill_value
    backscat = np.log10(backscat)

    arm_obj['backscatter'].data = backscat

    return arm_obj
