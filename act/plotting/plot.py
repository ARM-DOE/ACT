#Import third party libraries
import matplotlib.pyplot as plt
import datetime as dt
import astral
import numpy as np

#Import Local Libs
from . import common
from ..utils import datetime_utils as dt_utils


class display(object):
    def __init__(self,arm_obj):
        """Initialize Object"""
        self._arm = arm_obj
        self.fields = arm_obj.variables

        self.plots = []
        self.plot_vars = []
        self.cbs = []

    def day_night_background(self,ax=None,fig=None):
        #Get File Dates
        file_dates = self._arm.file_dates.data


        all_dates = dt_utils.dates_between(file_dates[-1],file_dates[0]) 

        #Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)

        # initialize the plot to a gray background for total darkness
        rect = ax.patch
        rect.set_facecolor('0.85')

        #Initiate Astral Instance
        a = astral.Astral()
        for f in all_dates:
            sun = a.sun_utc(f,self._arm.lat.data[0],
                self._arm.lon.data[0])

            # add yellow background for specified time period
            ax.axvspan(sun['sunrise'], sun['sunset'], facecolor='#FFFFCC')  
 
            #add local solar noon line
            ax.axvline(x=sun['noon'],linestyle='--', color='y')

    def plot(self,field,**kwargs):
        '''Function used to plot up data from the X-ARRAY dataset passed
           to it along with the corresponding features
           Keywords:
           xvariable - Variable names for the x-axis.  Defaults to time if none
           yvariable - Variable names for the y-axis.  Required
 
        '''
        data = self._arm[field].data
        xdim = list(self._arm[field].dims)
        xdata = self._arm[xdim[0]]

        ax = plt.gca()
        self.day_night_background()
        pm = ax.plot(xdata,data,'.')

        #Set X Limit
        xrng = [xdata.data[0],xdata.data[-1]]
        ax.set_xlim(xrng)

        #Set X Format
        days = (xrng[1]-xrng[0])/np.timedelta64(1, 'D')
        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)
