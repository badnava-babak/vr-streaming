from typing import Sequence

import numpy as np




def plot_traces(
        traces: np.ndarray,
        qp_labels: Sequence[int],
        *,
        ylabel: str,
        title: str,
):
    """Plot a (7, N) trace matrix.

    This helper keeps visualisation code out of notebooks if you prefer a
    functional style.
    """
    import matplotlib.pyplot as plt

    if qp_labels is None:
        qp_labels = list(range(7))

    frames = np.arange(traces.shape[1])
    plt.figure(figsize=(10, 4))
    for i, qp in enumerate(qp_labels):
        plt.plot(frames, traces[i], label=f"QP {qp}")

    # for x in range(0, traces.shape[1], 30):
    #     plt.axvline(x, linestyle="--", linewidth=0.8, alpha=0.4)
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    # plt.show()


def plot_qp_sizes(
        traces: np.ndarray,
        video_labels: Sequence[int],
        qp_labels,
        dividing_factor=1,
        *,
        ylabel: str,
        title: str,
        save=None
):

    import matplotlib.pyplot as plt

    if video_labels is None:
        video_labels = list(range(10))
    colors = [
    '#00BFFF',  # Deep Sky Blue
    '#3CB371',  # Medium Sea Green
]

    stats = []
    qp_levels = np.arange(7)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, video_id in enumerate(video_labels):
        data = traces[i] / dividing_factor

        mean = data.mean(axis=1)
        std = data.std(axis=1)
        p05 = np.quantile(data, 0.05, axis=1)
        p95 = np.quantile(data, 0.95, axis=1)
        median = np.quantile(data, 0.5, axis=1)

        stats.append(mean)

        ax.plot(qp_levels, mean, label=f"{video_id}", markersize=25, lw=3.5, color=colors[i])
        # if True:
        ax.fill_between(qp_levels, p05, p95, alpha=0.2, color=colors[i])
            # plt.plot(qp_levels, p05, color='tab:orange', linestyle="--", label="p5 / p95")
            # plt.plot(qp_levels, p95, color='tab:orange', linestyle="--")
            # median
            # plt.plot(qp_levels, median, color='tab:green', linewidth=1.5, label="median")

    ax.set_xlabel("QP Level", fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_xticks(ticks=qp_levels, labels=[str(qp) for qp in qp_labels])
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.5)
    # plt.legend(fontsize=18, framealpha=.6)
    if title:
        plt.title(title)
    ax.legend(fontsize=18, framealpha=.6, loc='upper left')
    ax.invert_xaxis()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300)
    plt.show()
    return stats



def plot_gr(
        traces: np.ndarray,
        video_labels: Sequence[int],
        qp_labels,
        *,
        ylabel: str,
        title: str,
):

    import matplotlib.pyplot as plt

    if video_labels is None:
        video_labels = list(range(10))

    g_r_values = []
    colors = [
        '#00BFFF',  # Deep Sky Blue
        '#3CB371',  # Medium Sea Green
    ]

    qp_levels = np.arange(7)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, video_id in enumerate(video_labels):
        tr = traces[i] / traces[i][0]
        # tr = (traces[i] - traces[i][0]) / traces[i][0]
        mean = tr.mean(axis=1)
        p05 = np.quantile(tr, 0.05, axis=1)
        p95 = np.quantile(tr, 0.95, axis=1)

        g_r_values.append((mean, tr.std(axis=1)))
        ax.plot(qp_levels, mean, label=f"{video_id}", lw=3.5, color=colors[i])
        ax.fill_between(qp_levels, p05, p95, alpha=0.2, color=colors[i])

    ax.invert_xaxis()


    ax.set_xlabel("QP Level", fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_xticks(ticks=qp_levels, labels=[str(qp) for qp in qp_labels])
    ax.grid()
    if title:
        plt.title(title)
    ax.legend(fontsize=18, framealpha=.6)
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    plt.show()
    return np.array(g_r_values)

