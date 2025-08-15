import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm


def plot_3d_scatter(df, metric):
    x = df['w0']
    y = df['w1']
    z = df['w2']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=df[metric], cmap='jet')
    # surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(df['latency_mean']), rstride=1, cstride=1, linewidth=0, antialiased=False)
    cbar = fig.colorbar(img)
    if metric == 'energy_mean':
        cbar.set_label("Average Energy Consumption (mW)")
    elif metric == 'latency_mean':
        cbar.set_label("Average Latency (s)")
    elif metric == 'psnr_mean':
        cbar.set_label("Average PSNR (dB)")
    elif metric == 'stall_time_mean':
        cbar.set_label("Average Stall Time (s)")

    ax.set_xlabel('PSNR Weight (W0)')
    ax.set_ylabel('Stall Time Weight (W1)')
    ax.set_zlabel('Energy Consumption Weight (W2)')


def plot_3d_objective_scatter(df, metric):
    x = df['w0']
    y = df['w1']
    z = df['w2']
    c = df['w0'] * df['psnr_mean']
    c -= df['w1'] * df['latency_mean']
    c -= df['w2'] * df['energy_mean']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=c, cmap='jet')
    # surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(df['latency_mean']), rstride=1, cstride=1, linewidth=0, antialiased=False)
    cbar = fig.colorbar(img)

    cbar.set_label("Objective Function")

    ax.set_xlabel('PSNR Weight (W0)')
    ax.set_ylabel('Stall Time Weight (W1)')
    ax.set_zlabel('Energy Consumption Weight (W2)')


def plot_w_vs_metric(df, w, metrics, multi_axis=True, save: str = None):
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']
    markers = ['-o', '-.*', '--x', ':s']
    fig, ax1 = plt.subplots()

    ax = ax1
    p = []
    for i, metric in enumerate(metrics):
        label = get_label(metric)

        data = (df
                .groupby([w])
                .agg(mean=(metric, 'mean'),
                     std=(metric, 'std'),
                     p5=(metric, lambda x: np.percentile(x, 5)),
                     p95=(metric, lambda x: np.percentile(x, 95))
                     )
                )
        p1, = ax.plot(data.index,
                      data['mean'], '-o',
                      color=colors[i],
                      alpha=0.85,
                      markersize=8,
                      # edgecolors="k",
                      linewidth=2., label=label.replace('Average ', ''))
        p.append(p1)
        # err_left, err_right = data['mean'] - data['energy_p5'], data['energy_p95'] - data['energy_mean']
        err_low, err_high = data['mean'] - data['p5'], data['p95'] - data['mean']
        ax.errorbar(data.index,
                    data['mean'],
                    # xerr=[[err_left], [err_right]],
                    yerr=[err_low.clip(0), err_high.clip(0)],
                    fmt="none",
                    ecolor=colors[i],
                    alpha=0.4,
                    capsize=3,
                    linewidth=1.)
        if multi_axis:
            ax.tick_params(axis='y', labelcolor=colors[i])
            ax.set_ylabel(label, color=colors[i])

        if multi_axis and i < len(metrics) - 1:
            ax = ax1.twinx()
            if i >= 1:
                ax.spines.right.set_position(("axes", 1. + i * .18))

    if not multi_axis:
        ax.set_ylabel("Fraction")
        ax.set_ylim(-.05, 1.05)
        ax1.grid()
    if w == 'w0':
        ax1.set_xlabel("PSNR weight ($W_0$)")
    elif w == 'w1':
        ax1.set_xlabel("Latency weight ($W_1$)")
    elif w == 'w2':
        ax1.set_xlabel("Energy Consumption weight ($W_2$)")
    fig.tight_layout()
    plt.legend(handles=p, loc='best', fontsize=12, framealpha=.6)
    if save:
        plt.savefig('results/figs/{}.pdf'.format(save), dpi=300)


def get_label(metric):
    if metric == 'obj':
        label = "Objective Value"
    if metric == 'energy_mean':
        label = "Average Energy Consumption (mW)"
    elif metric == 'latency_mean':
        label = "Average Latency (s)"
    elif metric == 'psnr_mean':
        label = "Average PSNR (dB)"
    elif metric == 'stall_time_mean':
        label = "Average Stall Time (s)"
    elif metric == '_5G_ratio':
        label = "5G Offloaded Tasks"
    elif metric == '_4G_ratio':
        label = "4G Offloaded Tasks"
    elif metric == '_WiGig_ratio':
        label = "WiGig Offloaded Tasks"
    elif metric == 'offload_ratio':
        label = "All Offloaded Task"
    elif metric == 'quality_6_ratio':
        label = "QP 35 Ratio"
    elif metric == 'quality_5_ratio':
        label = "QP 30 Ratio"
    elif metric == 'quality_4_ratio':
        label = "QP 25 Ratio"
    elif metric == 'quality_3_ratio':
        label = "QP 20 Ratio"
    elif metric == 'quality_2_ratio':
        label = "QP 15 Ratio"
    elif metric == 'quality_1_ratio':
        label = "QP 10 Ratio"
    elif metric == 'quality_0_ratio':
        label = "QP 5 Ratio"
    else:
        label = "Unknown Metric"

    qp_labels = [5, 10, 15, 20, 25, 30, 35]  # example QP values
    return label


def plot_contours(agg, metric):
    # levels = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]  # np.linspace(0, 2, 6)  # or np.linspace(0,10,6)
    levels = [1.]  # np.linspace(0, 2, 6)  # or np.linspace(0,10,6)
    for w2 in levels:
        slice_ = agg[agg['w2'] == w2].pivot(index="w0", columns="w1", values=metric)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        # plt.imshow(slice_, origin="lower", aspect="auto")
        x = df['w0'].unique()
        y = df['w1'].unique()
        x.sort()
        y.sort()
        img = ax.contourf(x, y, slice_, cmap='viridis')
        cbar = fig.colorbar(img)

        if metric == 'obj':
            cbar.set_label("Objective Value")
        if metric == 'energy_mean':
            cbar.set_label("Average Energy Consumption (mW)")
        elif metric == 'latency_mean':
            cbar.set_label("Average Latency (s)")
        elif metric == 'psnr_mean':
            cbar.set_label("Average PSNR (dB)")
        elif metric == 'stall_time_mean':
            cbar.set_label("Average Stall Time (s)")
        plt.title(f"w2 = {w2} (Latency weight)")

        plt.xlabel("w0  (PSNR weight)")
        plt.ylabel("w1  (Stall time weight)")
        # plt.colorbar(label="Mean latency (s)")
        plt.tight_layout()
    # plt.show()


def plot_pareto_set(agg, metrics, sense, plot_non_pareto=False):
    is_3d = len(metrics) > 2
    mask = paretoset(agg[metrics], sense=sense)
    pareto_front = agg[mask]
    non_pareto = agg[~mask]
    fig = plt.figure()
    opt_point = pareto_front.iloc[pareto_front['obj'].argmax()]
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')

        if plot_non_pareto:
            ax.scatter(non_pareto[metrics[0]], non_pareto[metrics[1]], non_pareto[metrics[2]],
                       c='b', s=50, alpha=0.2, label='None Pareto front')

        ax.scatter(pareto_front[metrics[0]], pareto_front[metrics[1]], pareto_front[metrics[2]],
                   c='r', s=50, alpha=0.5, label='Pareto front')

        ax.set_zlabel(get_label(metrics[2]))
    else:
        ax = fig.add_subplot(111)
        if plot_non_pareto:
            ax.scatter(non_pareto[metrics[0]], non_pareto[metrics[1]],
                       c='b', s=50, alpha=0.2, label='None Pareto front')
        ax.scatter(pareto_front[metrics[0]], pareto_front[metrics[1]],
                   c='r', s=50, alpha=0.5, label='Pareto front')
        ax.grid()

    # ax.scatter(opt_point['latency_mean'], opt_point['psnr_mean'], opt_point['energy_mean'], c='black',
    #             s=150, label=f"best $W_0$: {opt_point['w0']}, $W_1$: {opt_point['w1']}, $W_2$: {opt_point['w2']}")

    ax.set_xlabel(get_label(metrics[0]))
    ax.set_ylabel(get_label(metrics[1]))

    # plt.xlabel("Energy Consumption (mW)")
    plt.legend()


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

    df = pd.read_csv('results/exp-9.csv', index_col='seed')

    agg = (df
           .groupby(["w0", "w1", "w2"], as_index=False)
           .agg(latency_mean=("latency_mean", "mean"),
                stall_total=("stall_total", "mean"),
                psnr_mean=("psnr_mean", "mean"),
                energy_mean=("energy_mean", "mean"),
                stall_time_mean=("stall_time_mean", "mean"),
                offload_ratio=("offload_ratio", "mean"),
                _5G_ratio=("5G_ratio", "mean"),
                _4G_ratio=("4G_ratio", "mean"),
                _WiGig_ratio=("WiGig_ratio", "mean"),
                quality_0_ratio=("quality_0_ratio", "mean"),
                quality_1_ratio=("quality_1_ratio", "mean"),
                quality_2_ratio=("quality_2_ratio", "mean"),
                quality_3_ratio=("quality_3_ratio", "mean"),
                quality_4_ratio=("quality_4_ratio", "mean"),
                quality_5_ratio=("quality_5_ratio", "mean"),
                quality_6_ratio=("quality_6_ratio", "mean"),
                ))

    agg['obj'] = agg['w0'] * agg['psnr_mean']
    agg['obj'] -= agg['w1'] * agg['stall_time_mean']
    agg['obj'] -= agg['w2'] * agg['energy_mean']

    # df = df.query("w0 <= 2")
    # df = df.query("w1 <= 2")
    # df = df.query("w2 <= 2")
    # df = df.query("w0 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    # df = df.query("w1 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    # df = df.query("w2 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    # print(df.head())
    #
    # plot_3d_scatter(df, 'latency_mean')
    # plot_3d_scatter(df, 'psnr_mean')
    # plot_3d_scatter(df, 'energy_mean')
    # plot_3d_scatter(df, 'stall_time_mean')
    # plot_3d_objective_scatter(df, 'obj')
    # plt.show()
    from paretoset import paretoset

    plot_pareto_set(agg, ['latency_mean', 'psnr_mean', 'energy_mean'],
                    ['min', 'max', 'min'], True)
    plot_pareto_set(agg, ['latency_mean', 'psnr_mean', 'energy_mean'],
                    ['min', 'max', 'min'], False)
    plot_pareto_set(agg[agg['w2'] == 0.], ['latency_mean', 'psnr_mean'],
                    ['min', 'max'], True)
    plot_pareto_set(agg[agg['w0'] == 0.], ['latency_mean', 'energy_mean'],
                    ['min', 'min'], True)
    plot_pareto_set(agg, ['energy_mean', 'psnr_mean'],  # TODO run for w1= 0 as well
                    ['min', 'max'], True)
    # plot_pareto_set(agg, ['psnr_mean', 'energy_mean'], ['max', 'min'], True)
    plt.show()
    # plot_contours(agg, 'latency_mean')
    # plot_contours(agg, 'psnr_mean')
    # plot_contours(agg, 'energy_mean')
    # plot_contours(agg, 'stall_time_mean')
    plot_contours(agg, 'obj')
    # plt.show()

    plot_w_vs_metric(agg, 'w0', ['psnr_mean', 'latency_mean', 'energy_mean'],
                     save='performance_vs_w0')
    plot_w_vs_metric(agg, 'w1', ['psnr_mean', 'latency_mean', 'energy_mean'],
                     save='performance_vs_w1')
    plot_w_vs_metric(agg, 'w2', ['psnr_mean', 'latency_mean', 'energy_mean'],
                     save='performance_vs_w2')

    plot_w_vs_metric(agg, 'w0', ['offload_ratio', '_5G_ratio', '_4G_ratio', '_WiGig_ratio'],
                     False, save='action_vs_w0')
    plot_w_vs_metric(agg, 'w1', ['offload_ratio', '_5G_ratio', '_4G_ratio', '_WiGig_ratio'],
                     False, save='action_vs_w1')
    plot_w_vs_metric(agg, 'w2', ['offload_ratio', '_5G_ratio', '_4G_ratio', '_WiGig_ratio'],
                     False, save='action_vs_w2')

    plot_w_vs_metric(agg, 'w0',
                     [
                         # 'quality_6_ratio',
                         # 'quality_5_ratio',
                         # 'quality_4_ratio',
                         'quality_3_ratio',
                         'quality_2_ratio',
                         'quality_1_ratio',
                         'quality_0_ratio',
                     ], False, save='quality_vs_w0')
    plot_w_vs_metric(agg, 'w1',
                     [
                         # 'quality_6_ratio',
                         # 'quality_5_ratio',
                         # 'quality_4_ratio',
                         'quality_3_ratio',
                         'quality_2_ratio',
                         'quality_1_ratio',
                         'quality_0_ratio',
                     ], False, save='quality_vs_w1')

    plot_w_vs_metric(agg, 'w2',
                     [
                         # 'quality_6_ratio',
                         # 'quality_5_ratio',
                         # 'quality_4_ratio',
                         'quality_3_ratio',
                         'quality_2_ratio',
                         'quality_1_ratio',
                         'quality_0_ratio',
                     ], False, save='quality_vs_w2')

    plt.show()
