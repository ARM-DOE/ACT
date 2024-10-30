"""
Example plot using stripes
--------------------------

Plot up climate stripes plots from already
existing climatologies from ARM data.
Author: Adam Theisen

"""

import act
import matplotlib.pyplot as plt

# SGP E13 MET data has already been processed to yearly averages,
# removing data flagged by embedded qc and DQRs
url = 'https://raw.githubusercontent.com/AdamTheisen/ARM-Climatologies/refs/heads/main/results/sgpmetE13.b1_temp_mean_Y.csv'
col_names = ['time', 'temperature', 'count']
ds = act.io.read_csv(url, column_names=col_names, index_col=0, parse_dates=True)

# Drop years with less than 500000 samples
ds = ds.where(ds['count'] > 500000)

# Create plot display
display = act.plotting.TimeSeriesDisplay(ds, figsize=(10, 2))
reference = ['2003-01-01', '2013-01-01']
display.plot_stripes('temperature', reference_period=reference)

plt.show()
