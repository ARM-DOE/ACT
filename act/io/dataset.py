import xarray as xr

@xr.register_dataset_accessor('act')
class ACTAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.file_times = []
        self.file_dates = []
        self.datastream = None
        self.site = None
