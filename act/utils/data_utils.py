import numpy as np
import scipy.stats as stats
def add_in_nan(time,data):
    diff = time.diff(dim='time',n=1)/np.timedelta64(1, 's')
    mode = stats.mode(diff).mode[0]
    index = np.where(diff.values > 2.*mode)
    d_data = np.asarray(data)
    d_time = np.asarray(time)

    #for i in index[0]:
    #    t_diff = (d_time[i+1]-d_time[i-1])/2.
    #    t_diff = np.timedelta64(1,'s')
    #    d_time = np.insert(d_time,i,d_time[i-1]+t_diff)
    #    d_data = np.insert(d_data,i,-9999,axis=0)

    #    d_time = np.insert(d_time,i,d_time[i+1]+t_diff)
    #    d_data = np.insert(d_data,i,-9999,axis=0)

    return d_time,d_data
