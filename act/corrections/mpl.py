import numpy as np
import xarray as xr
import warnings


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
    print('Removing Heights <0')
    act = obj.act
    obj = obj.where(obj.height > 0)
    height = obj['height'].values

    # Get indices for calculating background
    print('Background')
    var_names = ['signal_return_co_pol','signal_return_cross_pol']
    ind = [obj.height.shape[1]-50,obj.height.shape[1]-2]

    # Subset last gates into new dataset 
    print('subset')
    dummy = obj.isel(range_bins=xr.DataArray(np.arange(ind[0],ind[1])))

    #obj.rename({'dim_0': 'range_bins'},inplace=True)
    obj.act = act

    #Turn off warnings
    warnings.filterwarnings("ignore")
 
    # Run through co and cross pol data for corrections
    for name in var_names:    
        print('BG Calc')
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
        op = obj['overlap_correction'].values[0,:,-1]
        op_height = obj['overlap_correction_heights'].values[0,:,-1]

        for j in range(len(obj['range_bins'].values)):
            print(j)
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
        print('Convert Data to Log')
        data = 10.*np.log10(data)

        # Write data to object
        print('Write to obj')
        obj[name].values = data

    # Create the co/cross ratio variable
    print('Ratio')
    ratio = (x_data/co_data)*100.
    print('Copy OBJ')
    obj['cross_co_ratio'] = obj[var_names[0]].copy(data=ratio)

    print('Done')
    return obj
