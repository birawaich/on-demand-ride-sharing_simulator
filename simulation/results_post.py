from metrics import MetricMaster
from metrics import MetricPlot
import pickle as pkl
import os
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import MultipleLocator
import scienceplots

import numpy as np
def plot_2d_data_spread(ax: Axes, timestamps, data, color: str,desc: str):
        # prepare data
        min_err = np.min(data, axis=0)         # shape: (n_timepoints,)
        max_err = np.max(data, axis=0)         # shape: (n_timepoints,)
        qlow = np.quantile(data, 0.05, axis=0)  # shape: (n_timepoints,)
        qhigh = np.quantile(data, 0.95, axis=0)
        mean = np.mean(data,axis=0)

        # plot
        ax.fill_between(timestamps, qlow, qhigh, alpha=0.1, label='5-95\% Quantile', color=color)
        ax.plot(timestamps, mean, label=f"Mean {desc}", color=color)
        ax.plot(timestamps, min_err, linestyle=':', color=color, alpha=0.6, label='Min/Max')
        ax.plot(timestamps, max_err, linestyle=':', color=color, alpha=0.6)

dir_path = os.path.dirname(os.path.realpath(__file__)) #get current directory
plt.style.use(['science','ieee'])
fleet_sizes = [1000, 2000, 3000]
fourCoverage = [0.50177421,0.925352,0.9625]
twoCoverage = [0.432004948,0.764479,0.90315]
fourTime = [267,180,108]
twoTime = [256,200,113]

fourCoverage_NB = [0.312488,0.53120,0.80420]
twoCoverage_NB =[0.30132586,0.4935506,0.632427]
fourTime_NB = [274,260,231]
twoTime_NB = [267,247,228]

fourCoverageref =[.6032,.937,.979]
twoCoverageref = [.412,.753,.942]
fourTimeref = [258.585,197.158,162.45]
twoTimeref = [325.38,288.845,191.746]

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
ax1:Axes = axes[0]
ax2:Axes = axes[1]
# --- Plot 1: Service Rate ---
ax1.plot(fleet_sizes, fourCoverageref, 'o-', label='Centralized (c(v) = 4)', color='black')
ax1.plot(fleet_sizes, fourCoverage, 's-', label='Distributed w/ Rebalancing (c(v) = 4)', color='blue')
ax1.plot(fleet_sizes, fourCoverage_NB, '^--', label='Distributed w/o Rebalancing (c(v) = 4)', color='red', alpha=0.7)

ax1.set_ylabel('Service Rate (\%)')
ax1.set_ylim(0, 1) # Set y-axis limits to better show the differences
ax1.set_title('Percentage of Requests Serviced')
ax1.legend(loc=8)
ax1.set_xticks(fleet_sizes)
ax1.grid(True, which='both', linestyle='--', linewidth=0.25)
ax1.set_xlabel('Fleet Size ($N_v$)')
# ax1.yaxis.set_major_locator(LogFormatter(10))
ax1.xaxis.set_ticks(fleet_sizes)
# --- Plot 2: Mean Wait Time ---
ax2.plot(fleet_sizes, fourTimeref, 'o-', label='Centralized (c(v) = 4)', color='black')
ax2.plot(fleet_sizes, fourTime, 's-', label='Distributed w/ Rebalancing (c(v) = 4)', color='blue')
ax2.plot(fleet_sizes, fourTime_NB, '^--', label='Distributed w/o Rebalancing (c(v) = 4)', color='red', alpha=0.7)

ax2.set_xlabel('Fleet Size ($N_v$)')
ax2.set_ylabel('Mean Wait Time (minutes)')
ax2.set_title('Average Passenger Wait Time')
ax2.legend(loc=8)
ax2.set_xticks(fleet_sizes)
ax2.grid(True, which='both', linestyle='--', linewidth=0.25)

# Set x-axis ticks to match the fleet sizes for clarity
ax2.set_xticks(fleet_sizes)

# Adjust layout to prevent titles/labels from overlapping
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
fig.tight_layout() #fix overlaps
fig.savefig(f"out/report1.png", dpi=300)
# plt.pause(0.01)
fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)
ax1:Axes = axes[0]
ax2:Axes = axes[1]
# --- Plot 1: Service Rate ---
ax1.plot(fleet_sizes, twoCoverageref, 'o-', label='Centralized (c(v) = 2)', color='black')
ax1.plot(fleet_sizes, twoCoverage, 's-', label='Distributed w/ Rebalancing (c(v) = 2)', color='blue')
ax1.plot(fleet_sizes, twoCoverage_NB, '^--', label='Distributed w/o Rebalancing (c(v) = 2)', color='red', alpha=0.7)

ax1.set_ylabel('Service Rate (\%)')
ax1.set_ylim(0, 1) # Set y-axis limits to better show the differences
ax1.set_title('Percentage of Requests Serviced')
ax1.legend(loc=8)
ax1.set_xticks(fleet_sizes)
ax1.grid(True, which='both', linestyle='--', linewidth=0.25)
ax1.set_xlabel('Fleet Size ($N_v$)')
# ax1.yaxis.set_major_locator(LogFormatter(10))
ax1.xaxis.set_ticks(fleet_sizes)
# --- Plot 2: Mean Wait Time ---
ax2.plot(fleet_sizes, twoTimeref, 'o-', label='Centralized (c(v) = 2)', color='black')
ax2.plot(fleet_sizes, twoTime, 's-', label='Distributed w/ Rebalancing (c(v) = 2)', color='blue')
ax2.plot(fleet_sizes, twoTime_NB, '^--', label='Distributed w/o Rebalancing (c(v) = 2)', color='red', alpha=0.7)

ax2.set_xlabel('Fleet Size ($N_v$)')
ax2.set_ylabel('Mean Wait Time (minutes)')
ax2.set_title('Average Passenger Wait Time')
ax2.legend(loc=8)
ax2.set_xticks(fleet_sizes)
ax2.grid(True, which='both', linestyle='--', linewidth=0.25)

# Set x-axis ticks to match the fleet sizes for clarity
ax2.set_xticks(fleet_sizes)

# Adjust layout to prevent titles/labels from overlapping
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
fig.tight_layout() #fix overlaps
fig.savefig(f"out/report2.png", dpi=300)
plt.show()