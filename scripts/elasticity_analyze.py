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
    markers = ['^', 'o', '*', 'v', 'd', '^', 'o', '*', 'v', 'd']
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

def load_raw_data(file_path):
    f = open(file_path, 'rb')
    ppg_data = pickle.load(f)
    df = pd.concat([pd.DataFrame(v) for v in ppg_data.values()])
    df['p_psnr'] = df['psnr'].copy()
    df.loc[df['latency'] > 1, 'p_psnr'] = 0
    return df

if __name__ == '__main__':
    # Define the new header

    data_raw = {
        f'$u_k = {7 - i}$':
            load_raw_data(f'results/elasticity-exp/w0_0.35_w1_0.85_w2_0.15/5u/PPG_{i}.pkl') for i in range(7)
    }
    data_raw.update({
        f'Elastic':
            load_raw_data(f'results/elasticity-exp/w0_0.35_w1_0.85_w2_0.15/5u/PPG.pkl')
    })

    # df = pd.read_csv('results/elasticity-exp/w0_0.35_w1_0.85_w2_0.15/stats.csv')
    # df_f = df[df['elastic'] == False]
    # overall_ = {f'$u_k = {7-i}$': df_f[df_f['elasticity_parameter'] == i].iloc[0] for i in range(1, 7)}
    # overall_.update({'Elastic': df[df['elastic']].iloc[0]})
    overall_ = data_raw


    for key, row in overall_.items():
        row_str = f"{key} & "
        row_str += f"${row['rewards'].mean():.2f}$ &"
        row_str += f"${row['psnr'].mean():.2f} \pm {row['psnr'].std():.2f}$ &"
        row_str += f"${row['p_psnr'].mean():.2f} \pm {row['p_psnr'].std():.2f}$ &"
        row_str += f"${row['latency'].mean():.2f} \pm {row['latency'].std():.2f}$ & "
        row_str += f"${row['energy_consumption'].mean():.2f} \pm {row['energy_consumption'].std():.2f}$ \\\\"
        print(row_str)


    # plot_x_vs_y(overall_, x_label='elasticity_parameter', y_label='psnr', error_bar=True)
    # plot_x_vs_y(overall_, x_label='elasticity_parameter', y_label='latency', error_bar=True)
    # plot_x_vs_y(overall_, x_label='elasticity_parameter', y_label='energy', error_bar=True)
    # plot_x_vs_y(overall_, x_label='elasticity_parameter', y_label='reward', error_bar=False, save='results/figs/scalibility.pdf')
    # plt.show()
    # print(overall_)
