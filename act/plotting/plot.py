import matplotlib.pyplot as plt
import datetime as dt


class display(object):
    def __init__(self,arm_obj):
        """Initialize Object"""
        self._arm = arm_obj
        self.fields = arm_obj.variables

        self.plots = []
        self.plot_vars = []
        self.cbs = []

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
        pm = ax.plot(xdata,data,'.')
 
