import matplotlib
matplotlib.use('Agg')

import act.io.armfiles as arm
import act.plotting.plot as armplot
import act.discovery.get_files as get_data

import glob
import matplotlib.pyplot as plt
import os

username='' #Add your credentials from the ARM live data web service
token='' # Accessed through the Data Discovery homepage
datastream = 'sgpmetE13.b1'
startdate = '2019-01-01'
enddate = '2019-01-07'

#get_data.download_data(username, token, datastream, startdate, enddate)

files = glob.glob(''.join(['./',datastream,'/*']))
met = arm.read_netcdf(files)
met_temp = met.temp_mean
met_rh = met.rh_mean
met_lcl = (20.+met_temp/5.)*(100.-met_rh)/1000.
met['met_lcl'] = met_lcl*1000.
met['met_lcl'].attrs['units'] = 'm'
met['met_lcl'].attrs['long_name'] = 'LCL Calculated from SGP MET E13'

cwd = os.getcwd()
met['met_lcl'].to_netcdf(path=cwd+'/met_lcl.nc',mode='w',engine='netcdf4')

display = armplot.display(met)
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(1,1,1)
display.plot('met_lcl',ax=ax)

datastream = 'sgpceilC1.b1'
#get_data.download_data(username, token, datastream, startdate, enddate)
files = glob.glob(''.join(['./',datastream,'/*']))
ceil = arm.read_netcdf(files)
display = armplot.display(ceil)
display.plot('first_cbh',ax=ax)

ax.legend()

fig.savefig('./vap.png')
