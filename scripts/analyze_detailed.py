import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.print_distance_table import prepare_data, load_raw_data
from src.commons.plots import get_label


def plot_psnr_vs_th(data, group, title, metric='psnr'):
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # ax = ax1.twinx()
    markers = ['^', 'o', '*', 'v', 'd']
    p = []
    i = -1
    for label, df in data.items():
        i += 1
        df['throughput'] = df['throughput']
        stats = df.groupby(by=group).agg(
            metric_mean=(metric, 'mean'),
            metric_std=(metric, 'std'),
            metric_p05=(metric, lambda x: np.percentile(x, 5)),
            metric_p95=(metric, lambda x: np.percentile(x, 95)),
            throughput_mean=('throughput', 'mean'),
            throughput_std=('throughput', 'std'),
            throughput_p05=('throughput', lambda x: np.percentile(x, 5)),
            throughput_p95=('throughput', lambda x: np.percentile(x, 95)),
            size_mean=('task_size', 'mean'),
            size_std=('task_size', 'std'),
            size_p05=('task_size', lambda x: np.percentile(x, 5)),
            size_p95=('task_size', lambda x: np.percentile(x, 95))
        )
        stats['throughput_mean'] = stats['throughput_mean'] / 1e9
        stats['throughput_p05'] = stats['throughput_p05'] / 1e9
        stats['throughput_p95'] = stats['throughput_p95'] / 1e9
        stats['size_mean'] = stats['size_mean'] / 1e9
        # p1 = ax1.scatter(stats['throughput_mean'], stats['psnr_mean'],
        #                  label=label, alpha=0.85, linewidth=2., marker=markers[i], s=150,)
        x_axis = 'size_mean' if group == 'size_segment' else 'throughput_mean'

        p1, = ax1.plot(stats[x_axis], stats[f'metric_mean'],
                       label=label, alpha=0.85, linewidth=2., marker=markers[i], ms=13,
                       linestyle='-', )

        met = 'psnr'
        p.append(p1)

    x_label = '$S(e_k)$ : Task Size (Gb)' if group == 'size_segment' else '$R_k(u_k)$ : Transfer Rate (Gbps)'
    ax1.set_xlabel(x_label, fontsize=18, fontweight='bold')
    ax1.set_ylabel(get_label(metric), fontsize=18, fontweight='bold')
    # ax.set_ylabel('Response Time (s)', fontsize=18, fontweight='bold')
    # ax.tick_params(axis='y', labelsize=18)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    plt.title(title)
    plt.legend(handles=p, loc='best', fontsize=12, framealpha=.6)
    plt.tight_layout()
    # plt.show()


def plot_decisions(df, x_metric, y_metric, title, offloading_decision):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ['Local', '5G', '4G', 'WiGig']
    q_labels = qp_labels = [5, 10, 15, 20, 25, 30, 35]  # example QP values
    if offloading_decision:
        for i in range(4):
            ax.scatter(df[df['offloading_decision'] == i][x_metric] / 1e9,
                       df[df['offloading_decision'] == i][y_metric], alpha=.4, label=labels[i])
    else:
        for i in range(7):
            ax.scatter(df[df['quality_decision'] == i][x_metric] / 1e9,
                       df[df['quality_decision'] == i][y_metric], alpha=.4, label=f"QP {q_labels[i]}")
    plt.legend()
    ax.set_xlabel(get_label(x_metric), fontsize=18, fontweight='bold')
    ax.set_ylabel(get_label(y_metric), fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    plt.title(title)
    plt.tight_layout()


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

    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='psnr')
    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='latency')
    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='energy_consumption')
    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='rewards')

    data_raw = {
        'Optimal': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/Optimal.pkl'),
        'CPPG': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/CPPG.pkl'),
        'PPG': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPG.pkl'),
        'EGreedy': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/EGreedy.pkl'),
        'PPO': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPO.pkl'),
    }

    for i in range(7):
        df_u1 = pd.DataFrame(data_raw['Optimal'][i])

        fig, ax = plt.subplots(5, figsize=(7, 6))
        ax[0].scatter(df_u1.index, df_u1['task_size']/1e9)

        ax[1].scatter(df_u1.index,df_u1['uplink_5g_rates']/1e9, label='Uplink 5G')
        ax[1].scatter(df_u1.index,df_u1['uplink_4g_rates']/1e9, label='Uplink 4G')
        ax[1].scatter(df_u1.index,df_u1['uplink_wigig_rates']/1e9, label='Uplink WiGig')

        ax[2].scatter(df_u1.index,df_u1['downlink_5g_rates']/1e9, label='Downlink 5G')
        ax[2].scatter(df_u1.index,df_u1['downlink_4g_rates']/1e9, label='Downlink 4G')
        ax[2].scatter(df_u1.index,df_u1['downlink_wigig_rates']/1e9, label='Downlink WiGig')

        ax[3].scatter(df_u1.index,df_u1['offloading_decision'])
        ax[4].scatter(df_u1.index,7-df_u1['quality_decision'])

        ax[0].set_ylabel('Task size')
        ax[1].set_ylabel('Throughput')
        ax[2].set_ylabel('Throughput')
        ax[3].set_ylabel('Offloading Decision')
        ax[4].set_ylabel('Quality Decision')

        ax[1].legend()
        plt.tight_layout()
    plt.show()

    data_all = {
        'Optimal': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/Optimal.pkl', False)[0],
        'CPPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/CPPG.pkl', False)[0],
        'PPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPG.pkl', False)[0],
        'EGreedy': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/EGreedy.pkl', False)[0],
        'PPO': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPO.pkl', False)[0],
    }

    plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='psnr')
    # plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='latency')
    # plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='energy_consumption')
    # plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='rewards')
    # plot_psnr_vs_th(data, group='size_segment', title='TX + RX', metric='rewards')
    # plt.show()

    df = data['PPG']

    x_metric = 'task_size'
    y_metric = 'psnr'
    title = 'Optimal'




    # Define metrics once
    metrics = {
        'mean': 'mean',
        'std': 'std',
        'p05': lambda x: np.percentile(x, 15),
        'p95': lambda x: np.percentile(x, 85),
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

    stats = data_all['PPG'].groupby(
        by=['offloading_decision']
    ).agg(
        **agg_dict
    ) / 1e9

    fig, ax = plt.subplots(figsize=(7, 6))
    ax1 = ax.twinx()
    labels = np.array(['Local', '5G', '4G', 'WiGig'])
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    x = np.array(list(range(4, 32, 8)))
    x = x[:len(stats.index)]
    x_ticks = labels[stats.index]
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
                 bottom=stats['size_p05'], label='Task Size', color='black')

    p5 = ax.bar(x + 1,
                stats['down_5g_p95'] - stats['down_5g_p05'],
                bottom=stats['down_5g_p05'], label=f"{labels[1]} DownLink", color=colors[0], alpha=.5)
    p6 = ax.bar(x + 2,
                stats['down_4g_p95'] - stats['down_4g_p05'],
                bottom=stats['down_4g_p05'], label=f"{labels[2]} DownLink", color=colors[1], alpha=.5)
    p7 = ax.bar(x + 3,
                stats['down_wigig_p95'] - stats['down_wigig_p05'],
                bottom=stats['down_wigig_p05'], label=f"{labels[3]} DownLink", color=colors[2], alpha=.5)

    plt.axvline(x=8, color='r', linestyle='--', linewidth=2)
    plt.axvline(x=16, color='r', linestyle='--', linewidth=2)

    ax.set_xticks(x, labels=x_ticks)
    ax.set_xlabel('Offloading Decision', fontsize=18, fontweight='bold')
    ax.set_ylabel('Throughput (Gbps)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Task Size (Gbps)', fontsize=18, fontweight='bold')
    ax1.set_ylim(0, 1.8)
    ax.set_ylim(0, 1.8)
    ax.tick_params(axis='both', labelsize=18)
    ax1.tick_params(axis='both', labelsize=18)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    plt.tight_layout()
    plt.legend(handles=[p1, p2, p3, p4, p5, p6, p7], loc='best', fontsize=12, framealpha=.6)
    plt.show()

    plot_decisions(data['Optimal'], 'throughput', 'psnr', 'Optimal', True)
    plot_decisions(data['PPG'], 'throughput', 'psnr', 'PPG', True)

    plot_decisions(data['Optimal'], 'throughput', 'psnr', 'Optimal', False)
    plot_decisions(data['PPG'], 'throughput', 'psnr', 'PPG', False)

    plot_decisions(data_all['Optimal'], 'task_size', 'psnr', 'Optimal', True)
    plot_decisions(data_all['PPG'], 'task_size', 'psnr', 'PPG', True)

    plot_decisions(data_all['Optimal'], 'task_size', 'psnr', 'Optimal', False)
    plot_decisions(data_all['PPG'], 'task_size', 'psnr', 'PPG', False)

    # plot_decisions(data['CPPG'], 'task_size', 'psnr', 'CPPG')
    # plot_decisions(data['EGreedy'], 'task_size', 'psnr', 'EGreedy')
    # plot_decisions(data['PPO'], 'task_size', 'psnr', 'PPO')

    plt.show()
