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
metrics1:MetricMaster = pkl.load(open(dir_path + "/../../data/2-2000NoRebalancingMetricsDouble.pkl", "rb"))
metrics2:MetricMaster = pkl.load(open(dir_path + "/../../data/2-2000RebalancingMetricsDouble.pkl", "rb"))
metrics1._metric_plot.plot_passenger_metrics()
metrics2._metric_plot.plot_passenger_metrics()
#save
metrics1._metric_plot.save_figure(outname="2-2000NoR1D")
metrics2._metric_plot.save_figure(outname="2-2000R1D")