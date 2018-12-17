#Import third party libraries
import matplotlib.pyplot as plt
import datetime as dt
import astral
import numpy as np

#Import Local Libs
from . import common
from ..utils import datetime_utils as dt_utils
from ..utils import data_utils 


class display(object):
    def __init__(self,arm_obj):
        """Initialize Object"""
        self._arm = arm_obj
        self.fields = arm_obj.variables
        self.ds = str(arm_obj['ds'].values)
        self.file_dates = arm_obj.file_dates.values

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
        if self._arm.lat.data.size > 1:
            lat = self._arm.lat.data[0]
            lon = self._arm.lon.data[0]
        else:
            lat = float(self._arm.lat.data)
            lon = float(self._arm.lon.data)

        for f in all_dates:
            sun = a.sun_utc(f,lat,lon)

            # add yellow background for specified time period
            ax.axvspan(sun['sunrise'], sun['sunset'], facecolor='#FFFFCC')  
 
            #add local solar noon line
            ax.axvline(x=sun['noon'],linestyle='--', color='y')

    def set_xrng(self,xrng,ax=None,fig=None):
        '''Set Xrange'''
        #Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)
        ax.set_xlim(xrng)
        self.xrng = xrng

    def set_yrng(self,yrng,ax=None,fig=None):
        '''Set Yrange'''
        #Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)
        ax.set_ylim(yrng)
        self.yrng = yrng

    def add_colorbar(self,mappable,title=None,ax=None,fig=None):
        #Get ax and fig for plotting
        ax, fig = common.parse_ax_fig(ax, fig)
        #Give the colorbar it's own axis so the 2D plots line up with 1D
        box = ax.get_position()
        pad, width = 0.01, 0.01
        cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
        cbar = plt.colorbar(mappable,cax=cax) 
        cbar.ax.set_ylabel(title, rotation=270)

    def plot(self,field,ax=None,fig=None,
        cmap=None,cbmin=None,cbmax=None,set_title=None,
        add_nan=False,**kwargs):
        '''Function used to plot up data from the X-ARRAY dataset passed
           to it along with the corresponding features
           Keywords:
           xvariable - Variable names for the x-axis.  Defaults to time if none
           yvariable - Variable names for the y-axis.  Required
 
        '''

        #Get data and dimensions
        data = self._arm[field]
        dim = list(self._arm[field].dims)
        xdata = self._arm[dim[0]]
        ytitle = ''.join(['(',data.attrs['units'],')'])
        if len(dim) > 1:
            ydata = self._arm[dim[1]]
            units = ytitle
            ytitle = ''.join(['(',ydata.attrs['units'],')'])
        else:
            ydata = None

        #Get the current plotting axis, add day/night background and plot data
        ax, fig = common.parse_ax_fig(ax, fig)

        if ydata is None:
            self.day_night_background()
            ax.plot(xdata,data,'.')
        else:
            #Add in nans to ensure the data are not streaking
            #if add_nan is True:
            #    xdata,data = data_utils.add_in_nan(xdata,data)
            mesh = ax.pcolormesh(xdata,ydata,data.transpose(),cmap=cmap,vmax=cbmax,
                vmin=cbmin)

        #Set Title
        if set_title is None:
            set_title = ' '.join([self.ds,field,'on',self.file_dates[0]])

        plt.title(set_title)
      
        #Set YTitle
        ax.set_ylabel(ytitle)

        #Set X Limit
        if hasattr(self,'xrng'):
            self.set_xrng(self.xrng)
        else:
            self.xrng = [xdata.data[0],xdata.data[-1]]
            self.set_xrng(self.xrng)

        #Set Y Limit
        if hasattr(self,'yrng'):
            self.set_yrng(self.yrng)

        #Set X Format
        days = (self.xrng[1]-self.xrng[0])/np.timedelta64(1, 'D')
        myFmt = common.get_date_format(days)
        ax.xaxis.set_major_formatter(myFmt)
  
        if ydata is not None:
            self.add_colorbar(mesh,title=units)
