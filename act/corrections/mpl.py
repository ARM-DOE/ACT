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

    Note: Deadtime and darkcount corrections are not being applied yet

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
        data = obj[name].where(obj[name] >0.).values
        
        for i in range(len(obj['time'].values)):
            data[i,:] = data[i,:]-signal_bg[i]

        # After Pulse Correction Variable
        mode = '_'.join(name.split('_')[-2:])
        ap = obj['afterpulse_correction_'+mode].values

        # Overlap Correction Variable
        op = obj['overlap_correction'].values[0,:,0]
        op_height = obj['overlap_correction_heights'].values[0,:,0]

        for j in range(len(obj['range_bins'].values)):
            # Afterpulse Correction
            data[:,j] = data[:,j] - ap[:,j]

            # R-Squared Correction
            data[:,j] = data[:,j] * height[:,j] ** 2.

            # Overlap Correction
            idx = (np.abs(op_height - height[0,j])).argmin()
            data[:,j] = data[:,j] * op[idx]

        #Get linear data for ratio
        if 'co_pol' in name:
           co_data = data
        if 'cross_pol' in name:
           x_data = data

        # Convert data to decibels
        data = 10.*np.log10(data)

        # Write data to object
        obj[name].values = data

    # Create the co/cross ratio variable
    ratio = (x_data/co_data)*100.
    #obj['cross_co_ratio'] = xr.DataArray(ratio)
    obj['cross_co_ratio'] = obj[var_names[0]].copy(data=ratio)

    return obj
