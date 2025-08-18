from matplotlib import pyplot as plt


def plot_x_vs_y(policy_performance_results, x_label, y_label, error_bar=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, stats in policy_performance_results.items():
        ax.scatter(stats[('%s_mean' % x_label)],
                   stats[('%s_mean' % y_label)],
                   alpha=0.85, edgecolors="k", linewidth=0.5, label=label)
        if error_bar:
            err_left = stats[('%s_mean' % x_label)] - stats[('%s_p05' % x_label)]
            err_right = stats[('%s_p95' % x_label)] - stats[('%s_mean' % x_label)]
            err_low = stats[('%s_mean' % y_label)] - stats[('%s_p05' % y_label)]
            err_high = stats[('%s_p95' % y_label)] - stats[('%s_mean' % y_label)]
            ax.errorbar(stats[('%s_mean' % x_label)],
                        stats[('%s_mean' % y_label)],
                        xerr=[[err_left], [err_right]],
                        yerr=[[err_low], [err_high]],
                        fmt="none",
                        ecolor="gray",
                        alpha=0.5,
                        capsize=3,
                        linewidth=0.8)

    if x_label == "energy":
        ax.set_xlabel("Avg. Energy consumption (mW)")
    elif x_label == "latency":
        ax.set_xlabel("Avg. Response Time (s)")
    elif x_label == "ymse":
        ax.set_xlabel("Avg. YMSE")
    elif x_label == "psnr":
        ax.set_xlabel("Avg. PSNR (dB)")
    elif x_label == "reward":
        ax.set_xlabel("Avg. Reward")

    if y_label == "psnr":
        ax.set_ylabel("Avg. PSNR (dB)")
    elif y_label == "latency":
        ax.set_ylabel("Avg. Response Time (s)")
    elif y_label == "ymse":
        ax.set_ylabel("Avg. YMSE")
    elif y_label == "reward":
        ax.set_ylabel("Avg. Reward")

    # ax.set_title(f"Energy vs PSNR")
    plt.legend()
    # plt.show()


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
    ax.set_xlabel("Mean energy consumption (mW)")
    ax.set_ylabel("Mean Response Time (s)")
    ax.set_title(f"Energy vs Response Time")
    plt.legend()
    plt.show()
