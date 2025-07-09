from matplotlib import pyplot as plt


def plot_energy_vs_psnr(policy_performance_results):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, stats in policy_performance_results.items():
        ax.scatter(stats['energy_mean'],
                   stats['psnr_mean'],
                   alpha=0.85, edgecolors="k", linewidth=0.5, label=label)
        err_left, err_right = stats['energy_p5'], stats['energy_p95']
        err_low, err_high = stats['psnr_p05'], stats['psnr_p95']
        ax.errorbar(stats['energy_mean'],
                    stats['psnr_mean'],
                    xerr=[[err_left], [err_right]],
                    yerr=[[err_low], [err_high]],
                    fmt="none",
                    ecolor="gray",
                    alpha=0.5,
                    capsize=3,
                    linewidth=0.8)
    ax.set_xlabel("Mean energy consumption (mW)")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title(f"Energy vs PSNR")
    plt.legend()
    plt.show()


def plot_energy_vs_latency(policy_performance_results):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, stats in policy_performance_results.items():
        ax.scatter(stats['energy_mean'],
                   stats['latency_mean'],
                   alpha=0.85, edgecolors="k", linewidth=0.5, label=label)
        err_left, err_right = stats['energy_p5'], stats['energy_p95']
        err_low, err_high = stats['latency_p05'], stats['latency_p95']
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
