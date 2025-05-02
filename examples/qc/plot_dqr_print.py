"""
Query the ARM DQR webservice
----------------------------

This example shows how to query the ARM Data Quality Report (DQR) webservice,
to retrieve the machine readable DQR information and print it out to screen.

Author: Adam Theisen
"""

import act

# Pass in a datastream, start date, and end date to get all DQRs in that range
# This will print it in a user readable format
dqr = act.qc.print_dqr('sgp30ecorE14.b1', '20150101', '20190101')

# If you want something that can be copy and pasted into a spreadsheet
# set pretty_print=False
dqr = act.qc.print_dqr('sgp30ecorE14.b1', '20150101', '20190101', pretty_print=False)
