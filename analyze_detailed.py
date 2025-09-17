import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from src.commons.plots import get_label
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import pickle
import matplotlib

from scipy.stats import wasserstein_distance


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_x_vs_y(policy_performance_results, x_label, y_label, save: str = None, error_bar=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ['^', 'o', '*', 'v']
    i = -1
    for label, stats in policy_performance_results.items():
        i += 1
        ax.scatter(stats[('%s' % x_label)],
                   stats[('%s' % y_label)],
                   alpha=0.25, edgecolors="k", linewidth=0.5, label=label, marker=markers[i])
        if error_bar:
            err_left = max(0, stats[('%s_mean' % x_label)] - stats[('%s_p05' % x_label)])
            err_right = max(0, stats[('%s_p95' % x_label)] - stats[('%s_mean' % x_label)])
            err_low = max(0, stats[('%s_mean' % y_label)] - stats[('%s_p05' % y_label)])
            err_high = max(0, stats[('%s_p95' % y_label)] - stats[('%s_mean' % y_label)])
            ax.errorbar(stats[('%s_mean' % x_label)],
                        stats[('%s_mean' % y_label)],
                        xerr=[[err_left], [err_right]],
                        yerr=[[err_low], [err_high]],
                        fmt="none",
                        ecolor="gray",
                        alpha=0.8,
                        markersize=15,
                        capsize=3,
                        linewidth=0.8)

    ax.set_xlabel(get_label('%s_mean' % x_label), fontsize=18, fontweight='bold')
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


    # df = pd.read_csv('results/ppg-exp/w0_0.8_w1_2.8_w2_0.8.csv')
    # df = pd.read_csv('results/ppg-exp/w0_1.0_w1_1.8_w2_0.2.csv')
    # df = pd.read_csv('results/ppg-exp/w0_0.41_w1_0.43_w2_0.13.csv')

    def plot_psnr_vs_th(data, group, title, metric='psnr'):

        fig, ax1 = plt.subplots(figsize=(7, 6))

        # ax = ax1.twinx()
        markers = ['^', 'o', '*', 'v']
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
            # ax.plot(stats['throughput_mean'], stats['latency_mean'], alpha=0.25, linewidth=2., )
            #
            # x_label = 'throughput'
            # y_label = 'psnr'
            # err_left = stats[('%s_mean' % x_label)] - stats[('%s_p05' % x_label)]
            # err_right = stats[('%s_p95' % x_label)] - stats[('%s_mean' % x_label)]
            # err_low = stats[('%s_mean' % y_label)] - stats[('%s_p05' % y_label)]
            # err_high = stats[('%s_p95' % y_label)] - stats[('%s_mean' % y_label)]
            # ax1.errorbar(stats[('%s_mean' % x_label)],
            #             stats[('%s_mean' % y_label)],
            #             xerr=[err_left, err_right],
            #             yerr=[err_low, err_high],
            #             fmt="none",
            #             ecolor="gray",
            #             alpha=0.8,
            #             markersize=15,
            #             capsize=3,
            #             linewidth=0.8)
            met = 'psnr'
            # met = 'task_size'
            # p1 = ax1.scatter(df[group] / 1e9, df[met], label=label, alpha=0.25, linewidth=2., )
            # p1 = ax1.scatter(df['comp_intensity'], df['task_size'], label=label, alpha=0.25, linewidth=2., )
            # ax.scatter(df[group], df['latency'], alpha=0.25, linewidth=2., )
            # ax.set_ylim(0, .4)
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


    def prepare_data(bins, file_path, only_offloaded=True):
        f = open(file_path, 'rb')
        ppg_data = pickle.load(f)
        df = pd.concat([pd.DataFrame(v) for v in ppg_data.values()])
        # df['throughput'] = (df['task_size'] + df['task_res_size']) / df['latency']
        df['throughput'] = (df['task_size'] + df['task_res_size']) / (df['tx_time'] + df['rx_time'])
        df['tx_throughput'] = df['task_size'] / df['tx_time']
        df['rx_throughput'] = df['task_res_size'] / df['rx_time']
        if only_offloaded:
            df = df[df['offloading_decision'] > 0]
            throughput_segment, ret_bins = pd.qcut(df['throughput'], q=bins, retbins=True)
            df['throughput_segment'] = throughput_segment.values.codes
        # else:
        #     df = df[df['offloading_decision'] == 0]
        # df['tx_segment'] = pd.cut(df['tx_throughput'], bins=bins, retbins=False).values.codes
        df['size_segment'] = pd.qcut(df['task_size'], q=bins, retbins=False).values.codes
        # df['rx_segment'] = pd.cut(df['rx_throughput'], bins=bins, retbins=False).values.codes

        # df['throughput_segment'] = pd.qcut(df['throughput'], q=bins, retbins=False).values.codes
        df['offloaded'] = (df['offloading_decision'] > 0).astype(int)
        df['5g'] = (df['offloading_decision'] == 1).astype(int)
        df['4g'] = (df['offloading_decision'] == 2).astype(int)
        df['wigig'] = (df['offloading_decision'] == 3).astype(int)
        for i in range(7):
            df[f'q_{i}'] = (df['quality_decision'] == i).astype(int)

        return df, bins


    file_path = 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl'
    bins = 3

    # opt, _ = prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl')
    data = {
        'Optimal': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl')[0],
        'CPPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl')[0],
        'PPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPG.pkl')[0],
        'EGreedy': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/EGreedy.pkl')[0],
    }


    plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='psnr')
    plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='latency')
    plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='energy_consumption')
    plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX', metric='rewards')


    data_all = {
        'Optimal': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl', False)[0],
        'CPPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl', False)[0],
        'PPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPG.pkl', False)[0],
        'EGreedy': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/EGreedy.pkl', False)[0],
    }

    # plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='psnr')
    # plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='latency')
    # plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='energy_consumption')
    plot_psnr_vs_th(data_all, group='size_segment', title='TX + RX', metric='rewards')
    plot_psnr_vs_th(data, group='size_segment', title='TX + RX', metric='rewards')
    plt.show()

    st = {}
    for label, df in data.items():
        stats = df.groupby(by='offloading_decision').count()
        # st = f"{label}: "
        st[label] = {}
        for i in range(4):
            if i in stats['latency'].keys():
                # st += f"{i} : {stats['latency'][i] / stats['latency'].sum():.2f}, "
                st[label][i] = stats['latency'][i] / stats['latency'].sum()
            else:
                st[label][i] = 0.
    st = pd.DataFrame(st)
    print('Offloading Policy Comparison')
    print(st)

    st = {}
    for label, df in data.items():
        stats = df.groupby(by='quality_decision').count()
        # st = f"{label}: "
        st[label] = {}
        for i in range(7):
            if i in stats['latency'].keys():
                # st += f"{i} : {stats['latency'][i] / stats['latency'].sum():.2f}, "
                st[label][i] = stats['latency'][i] / stats['latency'].sum()
            else:
                st[label][i] = 0.
    st = pd.DataFrame(st)
    print('Quality Policy Comparison')
    print(st)

    st = {}
    for label, df in data.items():
        stats = df.groupby(by=['quality_decision',
                               'offloading_decision']
                           ).agg(count=('latency', 'count'))
        st[label] = {}
        for i in range(7):
            st[label][i] = {}
            for j in range(4):
                st[label][i][j] = 0.
                if (i, j) in stats.index.to_list():
                    # st += f"{i} : {stats['latency'][i] / stats['latency'].sum():.2f}, "
                    st[label][i][j] = stats['count'].loc[(i, j)] / stats['count'].sum()
                else:
                    st[label][i][j] = 0.

    for label, s in st.items():
        df = pd.DataFrame(s)

        fig, ax = plt.subplots()
        im, cbar = heatmap(df, range(4), range(7), ax=ax, cmap="YlGn", cbarlabel="Action frequency")
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
        plt.title(label)
    plt.show()
    print(st)

    ll = []
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, df in data.items():
        df_a = df.groupby(by='size_segment').agg('mean')
        ax.scatter(df['uplink_5g_rates'], df['offloaded'], label=f'{label}', alpha=.85)
        metric = 'offloaded'
        ll.append(df_a[[metric]].rename(columns={metric: label}).T)
    plt.legend()
    plt.show()
    print(pd.concat(ll))

    d = {}
    ii = 0
    for i, df1 in st.items():
        p = pd.DataFrame(df1).to_numpy().flatten()
        d[i] = {}
        jj = 0
        for j, df2 in st.items():
            q = pd.DataFrame(df2).to_numpy().flatten()
            if jj >= ii:
                d[i][j] = .5 * np.abs(p - q).sum()
            else:
                d[i][j] = wasserstein_distance(p, q)
            jj += 1
        ii += 1
    print(pd.DataFrame(d))

    df = data['Optimal']
    plt.figure()
    # plt.hist(data['Optimal']['size_segment'])
    # plt.hist(data['Optimal'][data['Optimal']['offloaded'] == 0]['size_segment'])
    # plt.hist(df[df['offloaded'] == 0]['size_segment'])
    # plt.hist(df[df['offloaded'] > 0]['size_segment'])
    plt.hist(df[df['offloading_decision'] == 3]['uplink_wigig_rates'])
    plt.hist(df[df['offloading_decision'] == 3]['uplink_5g_rates'])
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 6))
    df = data['Optimal']
    # for label, df in data.items():
    # df_a = df[df['offloading_decision'] == a]
    # df_a = df.groupby(by='size_segment').agg('mean')

    ax.scatter(df['task_size'], df['offloaded'], label=label, alpha=.25)
    plt.legend(loc='best')
    plt.show()
    plot_x_vs_y(data, x_label='psnr', y_label='task_size', error_bar=False)
    plot_psnr_vs_th(data, group='throughput_segment', title='TX + RX')
    # plot_psnr_vs_th(data, group='tx_segment', title='TX')
    # plot_psnr_vs_th(data, group='rx_segment', title='RX')
    plt.show()
    ll = []
    for label, df in data.items():
        metric = 'psnr'
        stats = df.groupby(by='throughput_segment').agg(
            mean=(metric, 'mean'),
            std=(metric, 'std'),
            p5=(metric, lambda x: np.percentile(x, 5)),
            p95=(metric, lambda x: np.percentile(x, 95))
        )
        # print(f"{label}: {stats['throughput'] / 1e9}, {stats['psnr']}")
        ll.append(stats[[metric]].rename(columns={metric: label}).T)
    print(pd.concat(ll))
