from metrics import MetricMaster
from metrics import MetricPlot
import pickle as pkl
import os
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
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
# metrics1:MetricMaster = pkl.load(open(dir_path + "/../../data/NoRebalancingMetricsAug-11-2025_2331.pkl", "rb"))
# metrics1:MetricMaster = pkl.load(open(dir_path + "/../../data/RebalancingMetricsAug-12-2025_0021.pkl", "rb"))
# metrics1._metric_plot.plot_passenger_metrics()
#save
# metrics1._metric_plot.save_figure()
plt.style.use(['science','ieee'])
storage1 = pkl.load(open(dir_path + "/../../data/rebalancing_data_2-3000_nb.pkl", "rb"))
storage2 = pkl.load(open(dir_path + "/../../data/rebalancing_data_2-3000.pkl", "rb"))
storage1
# metrics1._metric_plot.plot_passenger_metrics()
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig.canvas.manager.set_window_title("Rebalancing Controller")
# get data
num_regions = len(storage1["spots"][0])
num_datapoints = len(storage1["spots"])
data_spots = np.empty((num_regions,num_datapoints))
data_timestamps = np.array(storage1["timestamps"])
for i, spots in enumerate(storage1["spots"]):
    data_spots[:,i] = spots
l1_errors_spots = np.abs(data_spots - np.mean(data_spots,axis=0))
# plot
plot_2d_data_spread(
    ax=ax, timestamps=data_timestamps, data=l1_errors_spots,
    color='blue', desc=f"Unbalanced L1 Error"
)
# get data
num_regions = len(storage2["spots"][0])
num_datapoints = len(storage2["spots"])
data_spots = np.empty((num_regions,num_datapoints))
data_timestamps = np.array(storage2["timestamps"])
for i, spots in enumerate(storage2["spots"]):
    data_spots[:,i] = spots
l1_errors_spots = np.abs(data_spots - np.mean(data_spots,axis=0))
# plot
plot_2d_data_spread(
    ax=ax, timestamps=data_timestamps, data=l1_errors_spots,
    color='red', desc=f"Balanced L1 Error"
)
# cosmetics
MetricPlot._set_axis_for_simtime(ax,which='x')
ax.set_ylabel('L1 Error')
ax.set_title("Evolution of Free Spot Spread over Regions")
ax.legend()
ax.grid(True)
fig.tight_layout() #fix overlaps
fig.savefig(f"out/rebalancing_controller_2-3000.png", dpi=300)
# plt.show(block=False)
plt.pause(0.01)
#show
# plt.show(block=False)

# input("Showing Plot of Environment Metrics. Press ENTER to continue.")

