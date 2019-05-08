import numpy as np
import xarray as xr


def correct_mpl(obj):
    """
    This procedure corrects MPL data
    1.) Throw out data before laser firing (heights < 0)
    2.) Remove background signal
    3.) Afterpulse Correction - Subtraction of (afterpulse-darkcount). 
        NOTE: Currently the Darkcount in VAPS is being calculated as 
        the afterpulse at ~30km. But that might not be absolutely 
        correct and we will likely start providing darkcount profiles 
        ourselves along with other corrections.
    4.) Range Correction
    5.) Overlap Correction (Multiply)

    Parameters
    ----------
    obj: Dataset object
        The ACT object. 

    Returns
    -------
    obj: Dataset object
        The ACT Object containing the corrected values.
    """

    # 1 - Remove negative height data
    obj = obj.where(obj.height > 0, drop=True)
    height = obj.height.values

    # 2 - Remove Background Signal
    var_names = ['signal_return_co_pol','signal_return_cross_pol']
    ind = [obj.height.shape[1]-50,obj.height.shape[1]-2]

    # Subset last gates into new dataset 
    dummy = obj.isel(range_bins=xr.DataArray(np.arange(ind[0],ind[1])))

    # Run through co and cross pol data for corrections
    for name in var_names:    
        signal = dummy[name]
        signal = signal.where(signal > -9998.)
        signal_bg = signal.mean(dim = 'dim_0').values
  
        # Seems to be the fastest way of removing background signal at the moment
        data = obj[name].values
        for i in range(len(obj['time'].values)):
            data[i,:] = data[i,:]-signal_bg[i]

        # After Pulse Correction Variable
        mode = '_'.join(name.split('_')[-2:])
        ap = obj['afterpulse_correction_'+mode].values

        # Overlap Correction Variable
        op = obj['overlap_correction'].values

        print(ap,np.shape(ap))
        for j in range(len(obj['height'].values)):
            # Afterpulse Correction
            data[:,j] = data[:,j] - ap[:,j]

            # R-Squared Correction
            data[:,j] = data[:,j] * height[:,j] ** 2.

            # Overlap Correction
        obj[name].values = data
 

    corrections=['afterpulse_correction','overlap_correction']

    

    return obj
