"""
Parse the ARM datastream filename
---------------------------------

This is an example of how to parse
the datastream filename into its constituent parts.

"""

from act.utils.data_utils import DatastreamParserARM

# Here we have a full path filename.
filename = '/data/sgp/sgpmetE13.b1/sgpmetE13.b1.20190501.024254.nc'

# What if we want to extract some metadata from the filename instead of reading the file
# and extracting from the global attributes. We can call the DatastreamParserARM() method
# and extract the string value from the object using its properties.

fn_obj = DatastreamParserARM(filename)
print(f"Site is {fn_obj.site}")
print(f"Datastream Class is {fn_obj.datastream_class}")
print(f"Facility is {fn_obj.facility}")
print(f"Level is {fn_obj.level}")
print(f"Datastream is {fn_obj.datastream}")
print(f"Date is {fn_obj.date}")
print(f"Time is {fn_obj.time}")
print(f"File extension is {fn_obj.ext}")

# We can also use the parser for just the datastream part to extract the parts.
# The other methods will not have a value and return None.

filename = 'sgpmetE13.b1'

fn_obj = DatastreamParserARM(filename)
print(f"\nSite is {fn_obj.site}")
print(f"Datastream Class is {fn_obj.datastream_class}")
print(f"Facility is {fn_obj.facility}")
print(f"Level is {fn_obj.level}")
print(f"Datastream is {fn_obj.datastream}")
print(f"Date is {fn_obj.date}")
print(f"Time is {fn_obj.time}")
print(f"File extension is {fn_obj.ext}")
