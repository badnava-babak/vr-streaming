import numpy as np, scipy.io as sio
from matplotlib import pyplot as plt
import pandas as pd
from sympy import hn1

from src.buffer_dynamics import simulate_buffer, plot_buffer_levels, find_max_fps, plot_stall_vs_fps, plot_stall_vs_fps_multi
from src.io import load_hn, load_rd
from src.plotting_utils import plot_traces, plot_qp_sizes
from src.viewport import build_fov_traces, aggregate_traces

hmd = load_hn('datasets/navigation/hn.mat')
rd = load_rd('datasets/navigation/rd.mat')

bitrate = rd['bitrate']  # length-15 list, each (7, 64, N)
ymse = rd['ymse']
bitdepth = rd['bitdepth']
qp_labels = [5, 10, 15, 20, 25, 30, 35]  # example QP values
print(hmd[0].shape)

video_names = ['Academic', 'Basketball', 'Bridge', 'GateNight', 'Runner',
               'SiyuanGate', 'SouthGate', 'StudyRoom', 'Sward', 'Chairlift',
               'Skateboard', 'Gaslamp', 'Harbor', 'KiteFlite', 'Trolley']

g_r_values = []

qp_levels = np.arange(7)
plt.figure(figsize=(10, 4))
for i, video_id in enumerate(video_names):
    tr = bitrate[i].sum(axis=2).sum(axis=1)
    tr = tr / tr[0]

    g_r_values.append(tr)
gr = pd.DataFrame(g_r_values)
print(gr.shape)
gr.to_csv('datasets/compression_factors.csv')
