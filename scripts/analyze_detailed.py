import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scripts.print_distance_table import prepare_data
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


def plot_decisions(df, x_metric, y_metric, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ['Local', '5G', '4G', 'WiGig']
    for i in range(4):
        ax.scatter(df[df['offloading_decision'] == i][x_metric] / 1e9,
                   df[df['offloading_decision'] == i][y_metric], alpha=.4, label=labels[i])
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
    data = {
        'Optimal': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl', False)[0],
        'CPPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl')[0],
        'PPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPG.pkl')[0],
        'EGreedy': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/EGreedy.pkl')[0],
        'PPO': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPO.pkl')[0],
    }

    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='psnr')
    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='latency')
    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='energy_consumption')
    # plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='rewards')

    data_all = {
        'Optimal': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl', False)[0],
        'CPPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl', False)[0],
        'PPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPG.pkl', False)[0],
        'EGreedy': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/EGreedy.pkl', False)[0],
        'PPO': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPO.pkl', False)[0],
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

    plot_decisions(data_all['Optimal'], 'task_size', 'psnr', 'Optimal')
    plot_decisions(data_all['PPG'], 'task_size', 'psnr', 'PPG')
    plot_decisions(data['CPPG'], 'task_size', 'psnr', 'CPPG')
    plot_decisions(data['EGreedy'], 'task_size', 'psnr', 'EGreedy')
    plot_decisions(data['PPO'], 'task_size', 'psnr', 'PPO')

    plt.show()


