import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from src.commons.plots import get_label
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import pickle
import matplotlib

from scipy.stats import wasserstein_distance


def plot_x_vs_y(policy_performance_results, x_label, y_label, save: str = None, error_bar=True, legend=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ['^', 'o', '*', 'd']
    colors = ['tab:blue', 'tab:orange', 'tab:green',  'tab:purple']
    ls = ['solid', 'dashed', 'dashdot', 'dotted', (0, (1, 10))]
    i = -1
    for label, stats in policy_performance_results.items():
        i += 1
        ax.plot(stats[('%s' % x_label)],
                stats[('%s_mean' % y_label)],
                alpha=1, linewidth=3.5, label=label, marker=markers[i], ms=13, ls=ls[i], color=colors[i])
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
    if legend:
        plt.legend(fontsize=18, framealpha=.6)

    if save:
        plt.savefig(save)

    # ax.set_title(f"Energy vs PSNR")

    # plt.show()

def load_raw_data(file_path):
    f = open(file_path, 'rb')
    ppg_data = pickle.load(f)
    df = pd.concat([pd.DataFrame(v) for v in ppg_data.values()])
    df['p_psnr'] = df['psnr'].copy()
    df.loc[df['latency'] > 1, 'p_psnr'] = 0
    return df

if __name__ == '__main__':

    exp_name = 'ppg-exp'
    stats = {'Optimal': [], 'IPPG': [], 'CPPG': [], 'PPO':[], 'EGreedy':[]}
    for nb in [2, 3, 4, 5, 6, 7, 8]:
        u = f'{nb}u'
        for method in ['Optimal', 'IPPG', 'CPPG', 'PPO', 'EGreedy']:
            # stats[method][-1]['num_users'] = nb
            df = load_raw_data(f'results/{exp_name}/w0_0.35_w1_0.85_w2_0.15/video-2/{u}/{method}.pkl')
            data = {'num_users': nb}
            for metric in ['p_psnr', 'latency', 'psnr', 'energy_consumption', 'rewards']:
                data[f'{metric}_mean'] = df[metric].mean()
                data[f'{metric}_p95'] = df[metric].quantile(q=0.95)
                data[f'{metric}_p05'] = df[metric].quantile(q=0.05)
            stats[method].append(data)

    data_all = {
        'Optimal': pd.DataFrame(stats['Optimal']),
        'IPPG': pd.DataFrame(stats['IPPG']),
        'CPPG': pd.DataFrame(stats['CPPG']),
        'EA-Offloader': pd.DataFrame(stats['PPO']),
        # 'EGreedy': pd.DataFrame(stats['EGreedy']),
    }

    # df = pd.read_csv('results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/video-2/stats-video-2.csv')
    # df['p_psnr'] = df['psnr'].copy()
    # df.loc[df['latency'] > 1, 'p_psnr'] = 0
    # df = df[df['num_users'].isin([2,3,4,5, 6,8])]


    # overall_ = {
    #     'Optimal': df[df['policy'] == 'Optimal'].sort_values(by='num_users'),
    #     'CPPG': df[df['policy'] == 'CPPG'].sort_values(by='num_users'),
    #     'IPPG': df[df['policy'] == 'PPG'].sort_values(by='num_users'),
    #     # 'EGreedy': df[df['policy'] == 'EGreedy'].sort_values(by='num_users'),
    #     'PPO': df[df['policy'] == 'PPO'].sort_values(by='num_users'),
    #     #     'Optimal Solution: 1 User': single_user_stats.summary_stats()['overall'],
    # }

    plot_x_vs_y(data_all, x_label='num_users', y_label='p_psnr', legend=False,
                error_bar=True, save='results/figs/scalibility_psnr.pdf')
    plot_x_vs_y(data_all, x_label='num_users', y_label='latency', legend=False,
                error_bar=True, save='results/figs/scalibility_latency.pdf')
    plot_x_vs_y(data_all, x_label='num_users', y_label='energy_consumption', error_bar=True, save='results/figs/scalibility_energy.pdf')
    plot_x_vs_y(data_all, x_label='num_users', y_label='rewards', legend=False,
                error_bar=False, save='results/figs/scalibility.pdf')
    plt.show()
    print(data_all)
