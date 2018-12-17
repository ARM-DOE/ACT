import numpy as np

def correct_ceil(arm_obj):
    backscat = arm_obj['backscatter'].data
    backscat[backscat <= 0] = 0.0000001
    backscat = np.log10(backscat)

    arm_obj['backscatter'].data = backscat

    return arm_obj
