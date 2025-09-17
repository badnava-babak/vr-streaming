import numpy as np, scipy.io as sio
from matplotlib import pyplot as plt
import pandas as pd
from sympy import hn1

from src.buffer_dynamics import simulate_buffer, plot_buffer_levels, find_max_fps, plot_stall_vs_fps, \
    plot_stall_vs_fps_multi
from src.commons.io import load_hn, load_rd, load_all_traces
from src.plotting.network_traces_utils import load_scenario_files, multi_stats, plot_multi_traces, plot_tx_times_per_qp, \
    plot_time_stats, load_net_trace
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

video_traces = {}
for video_id in range(10):
    video_traces[video_id] = build_fov_traces(bitrate[video_id], ymse[video_id], hmd[video_id][user_id],
                                              BITDEPTHS[video_id])
video_traces
video_names = ['Academic', 'Basketball',
               'Bridge', 'GateNight', 'Runner',
               'SiyuanGate',
               'SouthGate', 'StudyRoom', 'Sward', 'Chairlift',
               'Skateboard', 'Gaslamp', 'Harbor', 'KiteFlite', 'Trolley'
               ]

vid_idx = [9, 5, ]
labels = [video_names[i] for i in vid_idx]
video_bitrates = [video_traces[i]['bitrate'] for i in vid_idx]
video_psnr = [video_traces[i]['psnr'] for i in vid_idx]

colors = ['#00BFFF', '#3CB371']
fig, ax = plt.subplots(figsize=(7, 6))
for i in range(len(labels)):
    x = video_bitrates[i].flatten() / 1e9
    y = video_psnr[i].flatten()
    df = pd.DataFrame(np.array([x, y]).T, columns=['x', 'y'])
    bins = 12
    df['q'] = pd.qcut(df['x'], q=bins, labels=range(bins))
    df = df.groupby(by='q').agg(x=('x', "mean"), y=('y', 'mean'))
    plt.plot(df['x'], df['y'], color=colors[i], lw=3.5, label=labels[i])
    plt.scatter(x, y, color=colors[i], alpha=0.2)
ax.set_xlabel("Bitrate (Gbps)", fontsize=18, fontweight='bold')
ax.set_ylabel("PSNR (dB)", fontsize=18, fontweight='bold')
ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.legend(fontsize=18, framealpha=.6, loc='lower right')
ax.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig('results/figs/eda/bitrate_vs_psnr.pdf', dpi=300)
plt.show()



plot_qp_sizes(video_bitrates, labels, qp_labels,
              ylabel='Bitrate (Gbps)', title='', dividing_factor=1.e9,
              save='results/figs/eda/bit_rates.pdf')
plot_qp_sizes(video_psnr, labels, qp_labels,
              ylabel='PSNR (dB)', title='', save='results/figs/eda/psnr_stats.pdf')

gr_values = plot_gr(video_bitrates, labels, qp_labels, ylabel='Video Segment Size Increase', title='')
print(gr_values)

traces = build_fov_traces(bitrate[video_id], ymse[video_id], hmd[video_id][user_id],
                          BITDEPTHS[video_id])  # shape (7, N)

# -----------------------------------------------------------------
# 1. Point each scenario name to its file (CSV / TXT / TSV accepted)
# -----------------------------------------------------------------
_5g_data_path = "datasets/Network-Traces/Lumous5G/5G"
_4g_data_path = "datasets/Network-Traces/Lumous5G/4G"
_wigig_data_path = "datasets/Network-Traces/WiGig"
scenario_paths = {
    "5G-10": ("%s/5g_trace_10_walking" % _5g_data_path),
    "5G-11": ("%s/5g_trace_11_walking" % _5g_data_path),
    "5G-119-DR": ("%s/5g_trace_119_driving" % _5g_data_path),
    "4g-50112-a": ("%s/4g_trace_walking_50112_a" % _4g_data_path),
    "4g-50118-c": ("%s/4g_trace_walking_50118_c" % _4g_data_path),
    "4g-60044-dr": ("%s/4g_trace_driving_60044_dr" % _4g_data_path),
}

sc_traces = load_scenario_files(scenario_paths)  # OrderedDict[str, DataFrame]

import os


def load_all_traces2(net_type: str = '5G'):
    if net_type == '5G':
        data_path = "datasets/Network-Traces/Lumous5G/5G"
    elif net_type == '4G':
        data_path = "datasets/Network-Traces/Lumous5G/4G"
    elif net_type == 'WiGig':
        data_path = "datasets/Network-Traces/WiGig"
    else:
        raise Exception("Invalid Network Type")

    walking_traces_path = os.listdir(data_path)
    if net_type in ['4G', '5G']:
        walking_traces_path = list(filter(lambda x: 'walking' in x, walking_traces_path))
    walking_traces_path = ["%s/%s" % (data_path, f_name) for f_name in walking_traces_path]

    channels = {}
    for i, walking_trace in enumerate(walking_traces_path):
        try:
            net_df = load_net_trace(walking_trace)
            channels[i] = net_df
        except FileNotFoundError:
            # Generate synthetic mini traces if files are absent (demo only)
            # raise FileNotFoundError("Files not found.")
            continue

    return channels


channels_5g = load_all_traces2('5G')
# -----------------------------------------------------------------
# 2. Quick statistical table
# -----------------------------------------------------------------
print(multi_stats(sc_traces))  # duration, mean, p95, min / max …

# -----------------------------------------------------------------
# 3. Overlay throughput–time curves
# -----------------------------------------------------------------
# plot_multi_traces(sc_traces)  # vertical grid every 10 s


# seg_bitrate_Mbps  ← 1-D array for your chosen QP (one value per 1-s segment)
# net_df            ← DataFrame from load_net_trace() with time_ms + throughput_Mbps

colors = [
        '#00BFFF',  # Deep Sky Blue
        '#3CB371',  # Medium Sea Green
    ]
fig, ax = plt.subplots(figsize=(7, 6))
j = -1
for video in vid_idx:
    j += 1
    tx_times = np.zeros((len(channels_5g), 7))
    for i, channel in channels_5g.items():
        for q in range(7):
            tx = compute_tx_times(video_traces[video]['bitrate'][q, :], channel)
            tx_times[i, q] = tx.mean()

    p05 = np.quantile(tx_times, 0.05, axis=0)
    p95 = np.quantile(tx_times, 0.95, axis=0)

    tx_times = pd.DataFrame(tx_times).replace([np.inf, -np.inf], np.nan).dropna()

    ax.plot(qp_labels, np.mean(tx_times, axis=0), lw=3.5, label=video_names[video], color=colors[j])
    ax.fill_between(qp_labels, p05, p95, alpha=0.2, color=colors[j])

ax.set_xlabel("QP Level", fontsize=18, fontweight='bold')
ax.set_ylabel("Transmission time (s)", fontsize=18, fontweight='bold')
ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.5)
# plt.title("Segment download time vs. QP level")
ax.legend(fontsize=18, framealpha=.6, loc='upper left')
ax.tick_params(axis='both', labelsize=18)
ax.invert_xaxis()
plt.tight_layout()

plt.savefig('results/figs/eda/tx_times.pdf', dpi=300)
plt.show()

# plot_tx_times_per_qp(video_traces[0]['bitrate'], sc_traces["5G-11"], qp_labels=qp_labels)
# plot_tx_times_per_qp(traces['bitrate'], sc_traces["5G-119-DR"], qp_labels=qp_labels)

# 3.  Simulate buffer for all QPs
# buf_stats = simulate_buffer(traces['bitrate'], sc_traces["5G-10"],
#                             segment_duration_s=1.,
#                             initial_buffer_s=0.0,
#                             max_buffer_s=10.0,
#                             qp_labels=qp_labels)
#
# print("Total stall time per QP (s):", buf_stats['total_stall_s'])
#
# # 4.  Plot buffer evolution
# plot_buffer_levels(buf_stats, vline_every=None)
#
# # 3. Test for QP = 32 (index 2) allowing ≤3 % stall ratio
# max_ok_fps = find_max_fps(traces['bitrate'], sc_traces["5G-10"],
#                           qp_idx=0,
#                           fps_candidates=range(20, 91, 10),
#                           max_stall_ratio=0.1,
#                           init_buf=0.)
# print(f"Smoothest fps at QP under ≤3 % stalls:  {max_ok_fps} fps")

# for q in range(7):
#     plot_stall_vs_fps(traces['bitrate'], sc_traces["5G-10"],
#                       qp_idx=q,  # e.g. QP = 32
#                       fps_candidates=range(20, 51, 2),
#                       show_ratio=False,  # plot stall fraction
#                       color="tab:blue")

import os

files = os.listdir(_5g_data_path)

walking_traces_path = list(filter(lambda x: 'walking' in x, files))
driving_traces_path = list(filter(lambda x: 'driving' in x, files))

walking_traces_path = ["%s/%s" % (_5g_data_path, f_name) for f_name in walking_traces_path]

driving_traces_path = ["%s/%s" % (_5g_data_path, f_name) for f_name in driving_traces_path]

files_4g = os.listdir(_4g_data_path)
walking_4g_traces_path = list(filter(lambda x: 'walking' in x, files_4g))
walking_4g_traces_path = ["%s/%s" % (_4g_data_path, f_name) for f_name in files_4g]

files_wigig = os.listdir(_wigig_data_path)
walking_wigig_traces_path = ["%s/%s" % (_wigig_data_path, f_name) for f_name in files_wigig]

stats_df = plot_time_stats({
    '5G': walking_traces_path,
    '4G': walking_4g_traces_path,
    'WiGig': walking_wigig_traces_path
},
    bin_size_s=1.0,
    align_start=True,  # each trace starts at 0 s
    max_duration_s=120,  # compare first 5 min only
    title="",
    save='results/figs/eda/net_stats.pdf')

print(stats_df.head())  # numerical table if you need it
