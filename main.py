import numpy as np, scipy.io as sio
from matplotlib import pyplot as plt
import pandas as pd
from sympy import hn1

from src.buffer_dynamics import simulate_buffer, plot_buffer_levels, find_max_fps, plot_stall_vs_fps, plot_stall_vs_fps_multi
from src.commons.io import load_hn, load_rd
from src.plotting.network_traces_utils import load_scenario_files, multi_stats, plot_multi_traces, plot_tx_times_per_qp, \
    plot_time_stats
from src.plotting.plotting_utils import plot_traces, plot_qp_sizes, plot_gr
from src.plotting.viewport import build_fov_traces, compute_tx_times

hmd = load_hn('datasets/navigation/hn.mat')
rd = load_rd('datasets/navigation/rd.mat')

bitrate = rd['bitrate']  # length-15 list, each (7, 64, N)
ymse = rd['ymse']
hmd  # (12 videos × 12 users) list of Nx4

qp_labels = [5, 10, 15, 20, 25, 30, 35]  # example QP values
BITDEPTHS = np.array(
    [8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 8, 8, 8, 8], dtype=int)
video_id = 1
user_id = 0

traces = build_fov_traces(bitrate[video_id], ymse[video_id], hmd[video_id][user_id], BITDEPTHS[video_id])  # shape (7, N)
plot_traces(traces['bitrate'], qp_labels, ylabel='Bitrate [bps]', title='Bitrate Over Time')
plot_traces(traces['ymse'], qp_labels, ylabel='YMSE', title='YMSE Over Time')
plot_traces(traces['psnr'], qp_labels, ylabel='PSNR [dB]', title='PSNR Over Time')
plt.show()

plt.figure(figsize=(10, 4))
frames = np.arange(traces['bitrate'].shape[1])
plt.plot(qp_labels, traces['bitrate'].mean(axis=1))
plt.xlabel("QP ")
plt.ylabel("Bitrate [bps]")
plt.title("")
plt.legend()
plt.tight_layout()
plt.show()

video_traces = {}
for video_id in range(10):
    video_traces[video_id] = build_fov_traces(bitrate[video_id], ymse[video_id], hmd[video_id][user_id], BITDEPTHS[video_id])
video_traces
video_names = ['Academic', 'Basketball', 'Bridge', 'GateNight', 'Runner',
               'SiyuanGate', 'SouthGate', 'StudyRoom', 'Sward', 'Chairlift',
               'Skateboard', 'Gaslamp', 'Harbor', 'KiteFlite', 'Trolley']

video_bitrates = [video_traces[i]['bitrate'] for i in range(len(video_traces))]
plot_qp_sizes(video_bitrates, video_names[:10], qp_labels, ylabel='Bitrate [bps]', title='Bitrate Over Time')

gr_values = plot_gr(video_bitrates, video_names[:10], qp_labels, ylabel='Video Segment Size Increase', title='')
print(gr_values)


stats = pd.DataFrame({
    'QP': np.repeat(qp_labels, traces['bitrate'].shape[1]),
    'bitrate_Mbps': traces['bitrate'].ravel() / 1e6
})
print(stats.groupby('QP')['bitrate_Mbps']
      .agg(['mean', 'std', 'min', 'max', 'median']))


# df = load_net_trace('datasets/Network-Traces/Lumous5G/5G/5g_trace_10_walking')   # two-column CSV


# -----------------------------------------------------------------
# 1. Point each scenario name to its file (CSV / TXT / TSV accepted)
# -----------------------------------------------------------------
_5g_data_path = "datasets/Network-Traces/Lumous5G/5G"
_4g_data_path = "datasets/Network-Traces/Lumous5G/4G"
scenario_paths = {
    "5G-10": ("%s/5g_trace_10_walking" % _5g_data_path),
    "5G-11": ("%s/5g_trace_11_walking" % _5g_data_path),
    "5G-119-DR": ("%s/5g_trace_119_driving" % _5g_data_path),
    "4g-50112-a": ("%s/4g_trace_walking_50112_a" % _4g_data_path),
    "4g-50118-c": ("%s/4g_trace_walking_50118_c" % _4g_data_path),
    "4g-60044-dr": ("%s/4g_trace_driving_60044_dr" % _4g_data_path),
}

sc_traces = load_scenario_files(scenario_paths)  # OrderedDict[str, DataFrame]

# -----------------------------------------------------------------
# 2. Quick statistical table
# -----------------------------------------------------------------
print(multi_stats(sc_traces))  # duration, mean, p95, min / max …

# -----------------------------------------------------------------
# 3. Overlay throughput–time curves
# -----------------------------------------------------------------
plot_multi_traces(sc_traces)  # vertical grid every 10 s


# seg_bitrate_Mbps  ← 1-D array for your chosen QP (one value per 1-s segment)
# net_df            ← DataFrame from load_net_trace() with time_ms + throughput_Mbps
tx_times = compute_tx_times(video_traces[0]['bitrate'][0, :], sc_traces["5G-10"])

plot_tx_times_per_qp(video_traces[0]['bitrate'], sc_traces["5G-11"], qp_labels=qp_labels)
plot_tx_times_per_qp(traces['bitrate'], sc_traces["5G-119-DR"], qp_labels=qp_labels)

# 3.  Simulate buffer for all QPs
buf_stats = simulate_buffer(traces['bitrate'], sc_traces["5G-10"],
                            segment_duration_s=1.,
                            initial_buffer_s=0.0,
                            max_buffer_s=10.0,
                            qp_labels=qp_labels)

print("Total stall time per QP (s):", buf_stats['total_stall_s'])

# 4.  Plot buffer evolution
plot_buffer_levels(buf_stats, vline_every=None)

# 3. Test for QP = 32 (index 2) allowing ≤3 % stall ratio
max_ok_fps = find_max_fps(traces['bitrate'], sc_traces["5G-10"],
                          qp_idx=0,
                          fps_candidates=range(20, 91, 10),
                          max_stall_ratio=0.1,
                          init_buf=0.)
print(f"Smoothest fps at QP under ≤3 % stalls:  {max_ok_fps} fps")

for q in range(7):
    plot_stall_vs_fps(traces['bitrate'], sc_traces["5G-10"],
                      qp_idx=q,  # e.g. QP = 32
                      fps_candidates=range(20, 51, 2),
                      show_ratio=False,  # plot stall fraction
                      color="tab:blue")

import os

files = os.listdir(_5g_data_path)

walking_traces_path = list(filter(lambda x: 'walking' in x, files))
driving_traces_path = list(filter(lambda x: 'driving' in x, files))

walking_traces_path = ["%s/%s" % (_5g_data_path, f_name) for f_name in walking_traces_path]
driving_traces_path = ["%s/%s" % (_5g_data_path, f_name) for f_name in driving_traces_path]

# 3. Plot average stall curve for QP 32 (index 2)
plot_stall_vs_fps_multi(traces['bitrate'], walking_traces_path,
                        qp_idx=2,
                        initial_buffer_s=0.,
                        fps_candidates=range(20, 61, 5),
                        show_ratio=False,  # raw seconds
                        show_individual=True,  # faint per-trace lines
                        color="tab:blue")

stats_df = plot_time_stats(walking_traces_path,
                           bin_size_s=1.0,
                           align_start=True,  # each trace starts at 0 s
                           max_duration_s=300,  # compare first 5 min only
                           title="Throughput statistics over time")

print(stats_df.head())  # numerical table if you need it
