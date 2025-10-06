import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.print_distance_table import prepare_data, load_raw_data
from src.commons.plots import get_label


def plot_offloading_policy(df, title):
    stats = df.groupby(
        by=['offloaded']
    ).agg(
        **agg_dict
    ) / 1e9
    fig, ax = plt.subplots(figsize=(7, 6))
    ax1 = ax.twinx()
    labels = np.array(['Local', '5G', '4G', 'WiGig'])
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    colors = ['#32CD32',  # Lime Green
              '#404040',  # Black
              '#FF4500',  # Orange Red
              ]
    x = np.array(list(range(4, 32, 8)))
    x = x[:len(stats.index)]
    x_ticks = ['Local Computation', 'Offload Computation']
    w = 1
    p1 = ax.bar(x - 3,
                stats['up_5g_p95'] - stats['up_5g_p05'],
                bottom=stats['up_5g_p05'], label=f"{labels[1]} UpLink", color=colors[0])
    p2 = ax.bar(x - 2,
                stats['up_4g_p95'] - stats['up_4g_p05'],
                bottom=stats['up_4g_p05'], label=f"{labels[2]} UpLink", color=colors[1])
    p3 = ax.bar(x - 1,
                stats['up_wigig_p95'] - stats['up_wigig_p05'],
                bottom=stats['up_wigig_p05'], label=f"{labels[3]} UpLink", color=colors[2])
    p4 = ax1.bar(x,
                 stats['size_p95'] - stats['size_p05'],
                 bottom=stats['size_p05'], label='Task Size', color='#00BFFF')
    p5 = ax.bar(x + 1,
                stats['down_5g_p95'] - stats['down_5g_p05'],
                bottom=stats['down_5g_p05'], label=f"{labels[1]} DownLink", color=colors[0], alpha=.5)
    p6 = ax.bar(x + 2,
                stats['down_4g_p95'] - stats['down_4g_p05'],
                bottom=stats['down_4g_p05'], label=f"{labels[2]} DownLink", color=colors[1], alpha=.5)
    p7 = ax.bar(x + 3,
                stats['down_wigig_p95'] - stats['down_wigig_p05'],
                bottom=stats['down_wigig_p05'], label=f"{labels[3]} DownLink", color=colors[2], alpha=.5)
    plt.axvline(x=8, color='black', linestyle='--', linewidth=2, alpha=.3)
    plt.axhline(y=(stats.iloc[0]['size_p95'] + stats.iloc[1]['size_p05']) / 2,
                color='#00BFFF', linestyle='--', linewidth=2, alpha=.3)
    # plt.axhline(y=, color='r', linestyle='--', linewidth=2)
    # plt.axvline(x=16, color='r', linestyle='--', linewidth=2)
    ax.set_xticks(x, labels=x_ticks)
    ax.set_xlabel('Offloading Decision', fontsize=18, fontweight='bold')
    ax.set_ylabel('Throughput (Gbps)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Task Size (Gb)', fontsize=18, fontweight='bold')
    ax1.set_ylim(0, 1.8)
    ax.set_ylim(0, 1.8)
    ax.tick_params(axis='both', labelsize=16)
    ax1.tick_params(axis='both', labelsize=18)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    plt.tight_layout()
    plt.legend(ncols=2, handles=[p1, p2, p3, p4, p5, p6, p7], loc='upper right', fontsize=12, framealpha=.6)
    plt.title(title)
    plt.savefig('results/figs/offloading_policy.pdf')


if __name__ == '__main__':
    file_path = 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl'
    bins = 3

    # opt, _ = prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl')
    u = '8u'
    data = {
        'Optimal': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/Optimal.pkl')[0],
        'CPPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/CPPG.pkl')[0],
        'PPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPG.pkl')[0],
        'EGreedy': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/EGreedy.pkl')[0],
        'PPO': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPO.pkl')[0],
    }

    data_all = {
        'Optimal': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/Optimal.pkl', False)[0],
        'CPPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/CPPG.pkl', False)[0],
        'PPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPG.pkl', False)[0],
        'EGreedy': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/EGreedy.pkl', False)[0],
        'PPO': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPO.pkl', False)[0],
    }

    df = data['PPG']
    # df = df[~df['video_id'].isin([1, 8])]
    stats = df[df['offloading_decision'] > 0].groupby(by='throughput_segment').agg(['mean', 'std']).sort_values(
        by='throughput_segment')
    for index, row in stats.iterrows():
        # row_str = f"${row[('uplink_wigig_rates', 'mean')] / 1e9:.2f} \pm {row[('uplink_wigig_rates', 'std')] / 1e9:.2f}$ & "
        row_str = f"${row[('throughput', 'mean')] / 1e9:.2f} \pm {row[('throughput', 'std')] / 1e9:.2f}$ & "
        # row_str += f"${row[('downlink_wigig_rates', 'mean')] / 1e9:.2f} \pm {row[('downlink_wigig_rates', 'std')] / 1e9:.2f}$ & "
        # row_str += f"${row[('rewards', 'mean')]:.2f}\pm {row[('rewards', 'std')]:.2f}$ &"
        row_str += f"${row[('psnr', 'mean')]:.2f} \pm {row[('psnr', 'std')]:.2f}$ & "
        row_str += f"${row[('latency', 'mean')]:.2f} \pm {row[('latency', 'std')]:.2f}$ & "
        row_str += f"${row[('energy_consumption', 'mean')]:.2f} \pm {row[('energy_consumption', 'std')]:.2f}$ \\\\"
        print(index, row_str)

    print('\n\nVideo Stats\n')

    video_names = ['Academic', 'Basketball', 'Bridge', 'GateNight',
                   'Runner', 'SiyuanGate', 'SouthGate', 'StudyRoom', 'Sward']
    df = data_all['Optimal']
    stats = df.groupby(by='video_id').agg(['mean', 'std']).sort_values(by='video_id')
    for index, row in stats.iterrows():
        row_str = f"{video_names[index]} & ${row[('rewards', 'mean')]:.2f}$ & "
        row_str += f"${row[('psnr', 'mean')]:.2f} \pm {row[('psnr', 'std')]:.2f}$ &"
        row_str += f"${row[('latency', 'mean')]:.2f} \pm {row[('latency', 'std')]:.2f}$ &"
        row_str += f"${row[('energy_consumption', 'mean')]:.2f} \pm {row[('energy_consumption', 'std')]:.2f}$"
        print(row_str)
    stats_all = df.agg(['mean', 'std'])
    row_str = f"Total & ${stats_all.loc['mean']['rewards']:.2f}$ & "
    row_str += f"${stats_all.loc['mean']['psnr']:.2f} \pm {stats_all.loc['std']['psnr']:.2f}$ &"
    row_str += f"${stats_all.loc['mean']['latency']:.2f} \pm {stats_all.loc['std']['latency']:.2f}$ &"
    row_str += f"${stats_all.loc['mean']['energy_consumption']:.2f} \pm {stats_all.loc['std']['latency']:.2f}$ \\\\"
    print(row_str)

    # Define metrics once
    metrics = {
        'mean': 'mean',
        'std': 'std',
        'p05': lambda x: np.percentile(x, 5),
        'p95': lambda x: np.percentile(x, 95),
    }

    # Define which columns to apply metrics to
    cols = ['uplink_5g_rates', 'uplink_wigig_rates', 'uplink_4g_rates',
            'downlink_5g_rates', 'downlink_wigig_rates', 'downlink_4g_rates',
            'task_size']


    def rename(col):
        """Map raw column name to short prefix used in output."""
        if col.startswith("uplink_"):
            return "up_" + col.split("_")[1]  # e.g. uplink_5g_rates -> up_5g
        if col.startswith("downlink_"):
            return "down_" + col.split("_")[1]  # e.g. downlink_4g_rates -> down_4g
        if col == "task_size":
            return "size"
        return col


    # Build aggregation dictionary dynamically
    agg_dict = {
        f"{rename(col)}_{mname}": (col, func)
        for col in cols
        for mname, func in metrics.items()
    }
    df = data_all['Optimal']
    df['size_g_up_5g'] = df['task_size'] > df['uplink_5g_rates']
    df['size_g_up_wigig'] = df['task_size'] > df['uplink_wigig_rates']

    df['size_l_up_5g'] = df['task_size'] < df['uplink_5g_rates']
    df['size_l_up_wigig'] = df['task_size'] < df['uplink_wigig_rates']

    df['res_g_down_5g'] = df['task_res_size'] > df['downlink_5g_rates']
    df['res_g_down_wigig'] = df['task_res_size'] > df['downlink_wigig_rates']

    df['up_5g_g_wigig'] = df['uplink_5g_rates'] > df['uplink_wigig_rates']
    df['down_5g_g_wigig'] = df['downlink_5g_rates'] > df['downlink_wigig_rates']

    df['up_5g_l_wigig'] = df['uplink_5g_rates'] < df['uplink_wigig_rates']
    df['down_5g_l_wigig'] = df['downlink_5g_rates'] < df['downlink_wigig_rates']
    df = data_all['Optimal']

    plot_offloading_policy(data_all['Optimal'], 'Optimal')
    plot_offloading_policy(data_all['PPG'], 'IPPG')
    plot_offloading_policy(data_all['CPPG'], 'CPPG')
    plt.show()
