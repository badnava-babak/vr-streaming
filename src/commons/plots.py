from matplotlib import pyplot as plt


def get_label(metric):
    if metric == 'obj':
        label = "Objective Value"
    elif metric == 'rewards':
        label = "$\mathit{QTE}(\mathbf{e}, \mathbf{u})$ [Reward]"
    elif metric == 'reward_mean' or metric == 'rewards_mean':
        label = "Avg. Reward"
    elif metric == 'energy_mean' or metric == 'energy_consumption_mean':
        label = "Avg. Energy Consumption (mJoule)"
    elif metric == 'energy_consumption':
        label = "$E(\mathbf{e}, \mathbf{u})$ [Energy Consumption (mJoule)]"
    elif metric == 'latency_mean':
        label = "Avg. Response Time (s)"
    elif metric == 'latency':
        label = "$T(\mathbf{e}, \mathbf{u})$ [Latency (s)]"
    elif metric == 'psnr_mean':
        label = "Avg. PSNR (dB)"
    elif metric == 'p_psnr_mean':
        label = "Avg. Perceived PSNR (dB)"
    elif metric == 'psnr':
        label = "$Q(\mathbf{e})$ [PSNR (dB)]"
    elif metric == 'stall_time_mean':
        label = "Avg. Stall Time (s)"
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
    elif metric == 'num_users':
        label = "Number of Users"
    elif metric == 'task_size':
        label = "Task Size (Gb)"
    elif metric == 'throughput':
        label = "Transmission Rate (Gb)"
    else:
        label = "Unknown Metric"

    qp_labels = [5, 10, 15, 20, 25, 30, 35]  # example QP values
    return label


def plot_x_vs_y(policy_performance_results, x_label, y_label, save: str=None, error_bar=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ['^', 'o', '*', 'v', 'd']
    i = -1
    for label, stats in policy_performance_results.items():
        i += 1
        ax.scatter(stats[('%s_mean' % x_label)],
                   stats[('%s_mean' % y_label)],
                   alpha=0.85, edgecolors="k", s=250, linewidth=0.5, label=label, marker=markers[i])
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



def plot_x_vs_y2(policy_performance_results, x_label, y_label, save: str=None, error_bar=True, legend=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ['^', 'o', '*', 'v', 'd']
    i = -1
    for label, stats in policy_performance_results.items():
        i += 1
        ax.scatter(stats[x_label].mean(),
                   stats[y_label].mean(),
                   alpha=0.85, edgecolors="k", s=250, linewidth=0.5, label=label, marker=markers[i])
        if error_bar:
            err_left = max(0, stats[x_label].mean() - stats[x_label].quantile(0.05))
            err_right = max(0, stats[x_label].quantile(0.95) - stats[x_label].mean())
            err_low = max(0, stats[y_label].mean() - stats[y_label].quantile(0.05))
            err_high = max(0, stats[y_label].quantile(0.95) - stats[y_label].mean())
            ax.errorbar(stats[x_label].mean(),
                        stats[y_label].mean(),
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
    if legend:
        plt.legend(fontsize=18, framealpha=.6)

    if save:
        plt.savefig(save)
def plot_metric_distribution(policy_performance_results, metric):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, stats in policy_performance_results.items():
        stats = [p[metric] for p in stats]
        ax.scatter(stats['energy_mean'],
                   stats['latency_mean'],
                   alpha=0.85, edgecolors="k", linewidth=0.5, label=label)
        err_left, err_right = stats['energy_mean'] - stats['energy_p5'], stats['energy_p95'] - stats['energy_mean']
        err_low, err_high = stats['latency_mean'] - stats['latency_p05'], stats['latency_p95'] - stats['latency_mean']
        ax.errorbar(stats['energy_mean'],
                    stats['latency_mean'],
                    xerr=[[err_left], [err_right]],
                    yerr=[[err_low], [err_high]],
                    fmt="none",
                    ecolor="gray",
                    alpha=0.5,
                    capsize=3,
                    linewidth=0.8)
    ax.set_xlabel("Mean energy consumption (mW)", )
    ax.set_ylabel("Mean Response Time (s)")
    ax.set_title(f"Energy vs Response Time")
    plt.legend()
    plt.show()
