import xarray as xr


@xr.register_dataset_accessor('act')
class ACTAccessor(object):
    """
    The xarray accessor for ACT data structures. This adds functionality
    that includes storing the times and names of each file in the dataset.
    In addition, the datastream can be given a name and a site.

    This accessor is automatically registered when ACT is imported, so
    generally there is no need to register this yourself.

    Attributes
    ----------
    file_times : list of datetimes
        The list of times corresponding to each file in the dataset.
    file_dates : list of datetimes
        The list of dates corresponding to each file in the dataset.
    datastream : str
        The name of the datastream.
    site : str
        A string describing the name of the site.
    arm_standards_flag : ARMStandardsFlag
        An ARMStandardsFlag showing whether the dataset conforms to ARM
        standards. Will be None for datasets read by a generic reader.

    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.file_times = []
        self.file_dates = []
        self.datastream = None
        self.site = None
        self.arm_standards_flag = None
