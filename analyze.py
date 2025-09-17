import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

from src.commons.plots import plot_x_vs_y, get_label
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


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
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'tab:orange', 'tab:green']
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
        # ax.errorbar(data.index,
        #             data['mean'],
        #             # xerr=[[err_left], [err_right]],
        #             yerr=[err_low.clip(0), err_high.clip(0)],
        #             fmt="none",
        #             ecolor=colors[i],
        #             alpha=0.4,
        #             capsize=3,
        #             linewidth=1.)
        # ax.fill_between(data.index,
        #             data['p5'],
        #             data['p95'],
        #             color=colors[i],
        #             alpha=0.2,
        #             linewidth=1.)
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



def plot_contours(agg, metric):
    # levels = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]  # np.linspace(0, 2, 6)  # or np.linspace(0,10,6)
    levels = [.05]  # np.linspace(0, 2, 6)  # or np.linspace(0,10,6)
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


def plot_pareto_set(agg, metrics, sense, plot_non_pareto=False, save:str=None):
    is_3d = len(metrics) > 2
    mask = paretoset(agg[metrics], sense=sense)
    pareto_front = agg[mask]
    non_pareto = agg[~mask]
    fig, ax = plt.subplots(figsize=(7, 6))
    # opt_point = pareto_front.iloc[pareto_front['obj'].argmax()]
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')

        if plot_non_pareto:
            ax.scatter(non_pareto[metrics[0]], non_pareto[metrics[1]], non_pareto[metrics[2]],
                       c='b', s=50, alpha=0.2, label='None Pareto front')

        ax.scatter(pareto_front[metrics[0]], pareto_front[metrics[1]], pareto_front[metrics[2]],
                   c='r', s=50, alpha=0.5, label='Pareto front')

        ax.set_zlabel(get_label(metrics[2]))
        w = pareto_front[(pareto_front['w0'] != 0.0) & (pareto_front['w1'] != 0.0) & (pareto_front['w2'] != 0.0)]
        print(w[['w0', 'w1', 'w2']].mean())
    else:
        # ax = fig.add_subplot(111)
        if plot_non_pareto:
            ax.scatter(non_pareto[metrics[0]], non_pareto[metrics[1]],
                       c='dimgray', s=50, alpha=0.2, rasterized=True)
        # ax.scatter(pareto_front[metrics[0]], pareto_front[metrics[1]],
        #            c='r', s=50, alpha=0.5, label='Pareto front')
        data = pareto_front.sort_values(by=metrics[0])

        print(pareto_front[['w0', 'w1', 'w2']].mean())
        ax.plot(data[metrics[0]], data[metrics[1]],
                c='crimson', alpha=1., label='Pareto frontier', markersize=25, lw=5.5, rasterized=True)
        ax.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)
        # ax.set_frame_on(False)
        # ax.set_axis_off()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor('lightgray')
        ax.patch.set_alpha(0.5)
        ax.tick_params(length=0)


        # ax.scatter(opt_point['latency_mean'], opt_point['psnr_mean'], opt_point['energy_mean'], c='black',
    #             s=150, label=f"best $W_0$: {opt_point['w0']}, $W_1$: {opt_point['w1']}, $W_2$: {opt_point['w2']}")

        ax.set_xlabel(get_label(metrics[0]), fontsize=18, fontweight='bold')
        ax.set_ylabel(get_label(metrics[1]), fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=18)
        # plt.legend(fontsize=18, framealpha=.6)
        plt.tight_layout()


        if metrics == ['energy_mean', 'latency_mean']:
            # Create zoomed inset axes
            x1, x2, y1, y2 = 0.29, 0.36, 0.2, 0.38  # Define the region to zoom
            # axins = zoomed_inset_axes(ax, zoom=1.5, loc='lower right')  # Zoom in by 2.5, place in lower right
            axins = ax.inset_axes(
                [0.58, 0.05, 0.4, 0.3],
                xlim=(x1, x2), ylim=(y1, y2), )
            # axins.plot(x, y)
            axins.plot(data[metrics[0]], data[metrics[1]],
                       c='crimson', alpha=1., label='Pareto frontier', markersize=25, lw=5.5, rasterized=True)
            axins.scatter(non_pareto[metrics[0]], non_pareto[metrics[1]], c='dimgray', s=50, alpha=0.2, rasterized=True)

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            # Hide tick labels on the inset for a cleaner look
            # axins.set_xticks([])
            # axins.set_yticks([])

            # Mark the inset region on the main plot
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ls='dashed', ec="black", lw=1.5, alpha=0.8)  # Connect top-left of zoom box to bottom-right of inset
            axins.tick_params(axis='both', colors='black', labelsize=10)
            axins.grid(color='white', linestyle='-', linewidth=1, alpha=0.7)
            axins.set_facecolor('lightgray')
            axins.patch.set_alpha(0.5)
            axins.tick_params(length=0)
            # Get the x-axis tick labels and set their font weight to bold
            for label in axins.get_xticklabels():
                label.set_fontweight('bold')

            # Get the y-axis tick labels and set their font weight to bold
            for label in axins.get_yticklabels():
                label.set_fontweight('bold')
        else:
            # ax.text(1.5, 50.5, 'Example Text')
            ax.text(.38, 48., 'Pareto Frontier', fontsize=16, color='crimson', fontweight='bold', rotation=30)

    if save:
        plt.savefig(save, dpi=300)

    # plt.xlabel("Energy Consumption (mW)")


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
    df = pd.read_csv('results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/stats.csv')



    overall_ = {
        'Optimal': df[df['policy'] == 'Optimal'].iloc[0].to_dict(),
        'Centralized': df[df['policy'] == 'CPPG'].iloc[0].to_dict(),
        'Decentralized': df[df['policy'] == 'PPG'].iloc[0].to_dict(),
        'Epsilon Greedy': df[df['policy'] == 'EGreedy'].iloc[0].to_dict(),
        #     'Optimal Solution: 1 User': single_user_stats.summary_stats()['overall'],
    }

    for k, v in overall_.items():
        print(f"{k} &  {v['reward_mean']:.3f} & {v['psnr_mean']:.3f} & {v['latency_mean']:.3f} & {v['energy_mean']:.3f} & {v['deadline_violation']:.3f}")
    plot_x_vs_y(overall_, x_label='psnr', y_label='reward', error_bar=False)
    plot_x_vs_y(overall_, x_label='energy', y_label='psnr', save='results/figs/energy_vs_psnr.pdf')
    plot_x_vs_y(overall_, x_label='latency', y_label='psnr', save='results/figs/latency_vs_psnr.pdf')
    plot_x_vs_y(overall_, x_label='energy', y_label='latency', save='results/figs/energy_vs_latency.pdf')
    # plt.show()
    cols = 'reward_mean,latency_mean,latency_p95,latency_p05,energy_mean,energy_p05,energy_p95,psnr_mean,psnr_p05,psnr_p95,ymse_mean,ymse_p05,ymse_p95,stall_total,offload_ratio,5G_ratio,4G_ratio,WiGig_ratio,quality_0_ratio,quality_1_ratio,quality_2_ratio,quality_3_ratio,quality_4_ratio,quality_5_ratio,quality_6_ratio,stall_time_mean,stall_time_p05,stall_time_p95,policy,seed,verbose,num_episodes,num_users,video_id,user_id,device_proc_speed,device_cpu_freq,edge_proc_speed,weights,csv_log,w0,w1,w2'
    cols = cols.split(',')
    # df = pd.read_csv('results/exp-18.csv', names=cols)
    df = pd.read_csv('results/exp-21/exp-21.csv', names=cols)

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

    # plot_pareto_set(agg, ['latency_mean', 'psnr_mean', 'energy_mean'],
    #                 ['min', 'max', 'min'], True)
    # plot_pareto_set(agg, ['latency_mean', 'psnr_mean', 'energy_mean'],
    #                 ['min', 'max', 'min'], True)
    # plt.show()
    plot_pareto_set(agg, ['latency_mean', 'psnr_mean'],
                    ['min', 'max'], True, save='results/figs/latency_vs_psnr_pareto.pdf')
    plot_pareto_set(agg, ['energy_mean', 'latency_mean'],
                    ['min', 'min'], True, save='results/figs/latency_vs_energy_pareto.pdf')
    plot_pareto_set(agg, ['energy_mean', 'psnr_mean'],  # TODO run for w1= 0 as well
                    ['min', 'max'], True, save='results/figs/energy_vs_psnr_pareto.pdf')
    # plot_pareto_set(agg, ['psnr_mean', 'energy_mean'], ['max', 'min'], True)
    plt.show()
    # plot_contours(agg, 'latency_mean')
    # plot_contours(agg, 'psnr_mean')
    # plot_contours(agg, 'energy_mean')
    # plot_contours(agg, 'stall_time_mean')
    # plot_contours(agg, 'obj')
    # plt.show()

    plot_w_vs_metric(agg, 'w0', ['psnr_mean', 'latency_mean', 'energy_mean'], save='performance_vs_w0')
    # plot_w_vs_metric(agg, 'w1', ['psnr_mean', 'latency_mean', 'energy_mean'],
    #                  save='performance_vs_w1')
    # plot_w_vs_metric(agg, 'w2', ['psnr_mean', 'latency_mean', 'energy_mean'],
    #                  save='performance_vs_w2')
    #
    # plot_w_vs_metric(agg, 'w0', ['offload_ratio', '_5G_ratio', '_4G_ratio', '_WiGig_ratio'],
    #                  False, save='action_vs_w0')
    # plot_w_vs_metric(agg, 'w1', ['offload_ratio', '_5G_ratio', '_4G_ratio', '_WiGig_ratio'],
    #                  False, save='action_vs_w1')
    # plot_w_vs_metric(agg, 'w2', ['offload_ratio', '_5G_ratio', '_4G_ratio', '_WiGig_ratio'],
    #                  False, save='action_vs_w2')
    #
    # plot_w_vs_metric(agg, 'w0',
    #                  [
    #                      'quality_6_ratio',
    #                      'quality_5_ratio',
    #                      'quality_4_ratio',
    #                      'quality_3_ratio',
    #                      'quality_2_ratio',
    #                      'quality_1_ratio',
    #                      'quality_0_ratio',
    #                  ], False, save='quality_vs_w0')
    # plot_w_vs_metric(agg, 'w1',
    #                  [
    #                      'quality_6_ratio',
    #                      'quality_5_ratio',
    #                      'quality_4_ratio',
    #                      'quality_3_ratio',
    #                      'quality_2_ratio',
    #                      'quality_1_ratio',
    #                      'quality_0_ratio',
    #                  ], False, save='quality_vs_w1')
    #
    # plot_w_vs_metric(agg, 'w2',
    #                  [
    #                      'quality_6_ratio',
    #                      'quality_5_ratio',
    #                      'quality_4_ratio',
    #                      'quality_3_ratio',
    #                      'quality_2_ratio',
    #                      'quality_1_ratio',
    #                      'quality_0_ratio',
    #                  ], False, save='quality_vs_w2')

    plt.show()
