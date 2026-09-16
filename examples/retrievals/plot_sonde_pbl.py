"""
Planetary Boundary Layer Height Profile via Radiosonde Retrievals
----------------------------------------------------------

This examples shoes how to estimate the planetary boundary layer height
via the Liu-Liang 2010 technique, a Bulk Richardson Number method,
and the Heffter technique.

Author: Joe O'Brien
"""

from arm_test_data import DATASETS

import act

# Read the SGP radiosonde data for an example
filename_sonde = DATASETS.fetch('sgpsondewnpnC1.b1.20190101.053200.cdf')
sonde_ds = act.io.arm.read_arm_netcdf(filename_sonde)

# more explicit units are needed for Potential Temperature calculation
sonde_ds['tdry'].attrs['units'] = 'degree_Celsius'

# Estimate PBL Height via the Liu-Liang 2010 technique
act.retrievals.calculate_pbl_liu_liang(sonde_ds)
# Estimate PBL Height via the Bulk Richardson Number method
act.retrievals.calculate_pbl_bulk_richardson(sonde_ds)
# Estimate PBL Height via the Heffter technique
act.retrievals.calculate_pbl_heffter(sonde_ds)

# Calculate what pressure level corresponds to the PBL heights
pbl_heffter = sonde_ds.where(
    abs(sonde_ds.alt_ss - sonde_ds.pblht_heffter) < 1, drop=True
).atm_pres_ss.data[0]
pbl_liu = sonde_ds.where(
    abs(sonde_ds.alt_ss - sonde_ds.pblht_liu_liang) < 1, drop=True
).atm_pres_ss.data[0]
pbl_rich = sonde_ds.where(
    abs(sonde_ds.alt_ss - sonde_ds.pblht_bulk_richardson_pt25) < 1, drop=True
).atm_pres_ss.data[0]

# Display the Sonde PBL estimates over a Skew-T diagram
display = act.plotting.SkewTDisplay(sonde_ds)

display.plot_from_u_and_v(
    'u_wind',
    'v_wind',
    'pres',
    'tdry',
    'dp',
)
# narrowing in on lower atmosphere to highlight the PBL height estimates
display.set_yrng([1000, 500])
display.set_xrng([-30, 20])

display.axes[0].axhline(
    pbl_heffter, linestyle="--", color="tab:orange", linewidth=1, label="Heffter PBL Height"
)
display.axes[0].axhline(
    pbl_liu, linestyle="--", color="tab:blue", linewidth=1, label="Liu & Liang PBL Height"
)
display.axes[0].axhline(
    pbl_rich, linestyle="--", color="tab:green", linewidth=1, label="Bulk Richardson PBL Height"
)
display.axes[0].legend(loc="upper right", fontsize=8)
