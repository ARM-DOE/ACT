# Place python module imports here, example:
import os
import matplotlib.pyplot as plt
import act

# Place arm username and token or example file if username and token
# aren't set, example:
username = os.getenv('ARM_USERNAME')
token = os.getenv('ARM_PASSWORD')

# Download and read file or files with the IO and discovery functions
# within ACT, example:
results = act.discovery.download_arm_data(
    username, token, 'sgpceilC1.b1', '2022-01-14', '2022-01-19'
)
ceil_ds = act.io.arm.read_arm_netcdf(results)

# Plot file using the ACT display submodule, example:
display = act.plotting.TimeSeriesDisplay(ceil_ds)
display.plot('backscatter')
plt.show()
