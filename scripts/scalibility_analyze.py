import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from src.commons.plots import get_label
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import pickle
import matplotlib

from scipy.stats import wasserstein_distance

def plot_x_vs_y(policy_performance_results, x_label, y_label, save: str = None, error_bar=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ['^', 'o', '*', 'v', 'd']
    i = -1
    for label, stats in policy_performance_results.items():
        i += 1
        ax.plot(stats[('%s' % x_label)],
                   stats[('%s_mean' % y_label)],
                   alpha=1, linewidth=3.5, label=label, marker=markers[i], ms=13)
        if error_bar:
            # err_left = max(0, stats[('%s_mean' % x_label)] - stats[('%s_p05' % x_label)])
            # err_right = max(0, stats[('%s_p95' % x_label)] - stats[('%s_mean' % x_label)])
            err_low = stats[('%s_mean' % y_label)] - stats[('%s_p05' % y_label)]
            err_high = stats[('%s_p95' % y_label)] - stats[('%s_mean' % y_label)]
            ax.errorbar(stats[('%s' % x_label)],
                        stats[('%s_mean' % y_label)],
                        # xerr=[[err_left], [err_right]],
                        yerr=[err_low.clip(0), err_high.clip(0)],
                        fmt="none",
                        ecolor="gray",
                        alpha=0.8,
                        markersize=15,
                        capsize=3,
                        linewidth=0.8)

    ax.set_xlabel(get_label('%s' % x_label), fontsize=18, fontweight='bold')
    ax.set_ylabel(get_label('%s_mean' % y_label), fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    plt.tight_layout()
    plt.legend(fontsize=18, framealpha=.6)

    if save:
        plt.savefig(save)

    # ax.set_title(f"Energy vs PSNR")

    # plt.show()


if __name__ == '__main__':
    # Define the new header
    new_header_list = ['policy', 'seed', 'num_users', 'video_id',
                       'user_id', 'device_proc_speed', 'device_cpu_freq',
                       'edge_proc_speed', 'w0', 'w1', 'w2', 'csv_log',
                       'latency_mean', 'latency_p95', 'latency_p05',
                       'energy_mean', 'energy_p5', 'energy_p95',
                       'psnr_mean', 'psnr_p05', 'psnr_p95',
                       'ymse_mean', 'ymse_p05', 'ymse_p95',
                       'stall_total',
                       'offload_ratio', '5G_ratio', '4G_ratio', 'WiGig_ratio']

    df = pd.read_csv('../results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/stats.csv')
    df = df[df['num_users'].isin([5, 6, 7, 8,  12])]
    overall_ = {
        'Optimal': df[df['policy'] == 'Optimal'].sort_values(by='num_users'),
        'CPPG': df[df['policy'] == 'CPPG'].sort_values(by='num_users'),
        'IPPG': df[df['policy'] == 'PPG'].sort_values(by='num_users'),
        'EGreedy': df[df['policy'] == 'EGreedy'].sort_values(by='num_users'),
        'PPO': df[df['policy'] == 'PPO'].sort_values(by='num_users'),
        #     'Optimal Solution: 1 User': single_user_stats.summary_stats()['overall'],
    }

    plot_x_vs_y(overall_, x_label='num_users', y_label='psnr', error_bar=True)
    plot_x_vs_y(overall_, x_label='num_users', y_label='latency', error_bar=True)
    plot_x_vs_y(overall_, x_label='num_users', y_label='energy', error_bar=True)
    plot_x_vs_y(overall_, x_label='num_users', y_label='reward', error_bar=False, save='results/figs/scalibility.pdf')
    plt.show()
    print(overall_)