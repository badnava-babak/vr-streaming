import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.network_traces_utils import compute_tx_times, load_net_trace


# --------------------------------------------------------------------------- #
# 1.  Buffer-dynamics simulator
# --------------------------------------------------------------------------- #
def simulate_buffer(
        seg_bitrate_Mbps: np.ndarray,  # (7, T)  from build_fov_traces()['bitrate']
        net_df: pd.DataFrame,  # columns  time_ms, throughput_Mbps
        segment_duration_s: float = 1.0,
        initial_buffer_s: float = 2.0,
        max_buffer_s: float = 10.0,
        qp_labels=None):
    """
    Simulate headset playback buffer for all QP levels.

    Returns
    -------
    dict  with keys 'buffer' (7×T array), 'stall' (7×T array, bool),
                      'total_stall_s' (7-element list)
    """
    if qp_labels is None:
        qp_labels = list(range(seg_bitrate_Mbps.shape[0]))

    T = seg_bitrate_Mbps.shape[1]
    buf_traces = np.zeros((7, T))  # buffer level at each segment start (s)
    stall_flags = np.zeros((7, T), bool)  # stall started before segment plays?
    total_stall = np.zeros(7)  # cumulative stall time

    # Pre-compute per-segment download times for every QP
    tx_times = np.array([compute_tx_times(seg_bitrate_Mbps[q], net_df,
                                          seg_duration_s=segment_duration_s)
                         for q in range(7)])  # shape (7, T)

    for q in range(7):
        buf = initial_buffer_s  # current buffer level (s)
        cur_time = 0.0  # wall-clock time (s)
        for t in range(T):
            # If buffer is empty we are stalled until this seg finishes downloading
            if buf <= 0:
                stall_flags[q, t] = True

            # Download segment t
            dl_time = tx_times[q, t]
            start_dl = cur_time
            end_dl = cur_time + dl_time

            # During download, playback drains the buffer
            drain = dl_time
            if buf > 0:
                consumed = min(buf, drain)
                buf -= consumed
                stall_time = drain - consumed
            else:
                stall_time = drain

            total_stall[q] += stall_time

            # Segment arrives → buffer increases by segment_duration_s
            buf = min(buf + segment_duration_s, max_buffer_s)

            # Post-download bookkeeping
            buf_traces[q, t] = buf
            cur_time = end_dl

    return {
        'buffer': buf_traces,
        'stall': stall_flags,
        'total_stall_s': total_stall,
        'qp_labels': qp_labels
    }


# --------------------------------------------------------------------------- #
# 2.  Plot helper
# --------------------------------------------------------------------------- #
def plot_buffer_levels(buffer_dict, vline_every=None, **plot_kwargs):
    """
    Visualise buffer level for every QP on a single figure.
    buffer_dict is the output of simulate_buffer().
    """
    buf = buffer_dict['buffer']
    qp_labels = buffer_dict['qp_labels']
    T = buf.shape[1]
    x = np.arange(T)

    plt.figure(figsize=(10, 4))
    for q in range(7):
        plt.plot(x, buf[q], label=f"QP {qp_labels[q]}", **plot_kwargs)

    if vline_every:
        for x0 in range(0, T, vline_every):
            plt.axvline(x0, ls='--', alpha=0.3)

    plt.xlabel("Segment index ({} s each)".format(int(round(plot_kwargs.get("segment_duration_s", 1)))))
    plt.ylabel("Buffer level [s]")
    plt.title("Headset buffer level vs. time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def rescale_bitrate(seg_bitrate_Mbps, target_fps, base_fps=30):
    """Scale a (7,T) bitrate matrix from base_fps to target_fps."""
    return seg_bitrate_Mbps * (target_fps / base_fps)


def find_max_fps(seg_bitrate_Mbps, net_df, qp_idx=0,
                 fps_candidates=range(30, 60, 120),  # 15 … 60 fps
                 max_stall_ratio=0.05,  # ≤5 % of play time
                 init_buf=2.0, max_buf=10.0):
    """
    Returns the highest FPS whose rebuffering ratio ≤ max_stall_ratio.
    """
    best_fps = None
    for fps in fps_candidates:
        # rate_scaled = rescale_bitrate(seg_bitrate_Mbps[[qp_idx]], fps)[0]

        rate_scaled = np.array([rescale_bitrate(seg_bitrate_Mbps[q], fps) for q in range(7)])

        # compute download times once for speed
        buf = simulate_buffer(rate_scaled, net_df,
                              initial_buffer_s=init_buf,
                              max_buffer_s=max_buf)
        stall = buf['total_stall_s'][qp_idx]
        video_time = len(rate_scaled)  # seconds (1 segment = 1 s)
        if stall / video_time <= max_stall_ratio:
            best_fps = fps  # still smooth – keep searching upward
        else:
            break  # first “bad” fps encountered
    return best_fps


def plot_stall_vs_fps(seg_bitrate_Mbps,
                      net_df,
                      qp_idx=0,
                      fps_candidates=range(15, 91, 5),
                      segment_duration_s=1.0,
                      initial_buffer_s=0.0,
                      max_buffer_s=10.0,
                      show_ratio=False,
                      **plot_kwargs):
    """
    Stall-time sweep: download the same video at different frame-rates and
    visualise how total rebuffering changes.

    Parameters
    ----------
    seg_bitrate_Mbps : (7, T) ndarray
        Per-segment FoV-weighted bitrate (30 fps baseline) from build_fov_traces().
    net_df : DataFrame
        Network log with columns time_ms, throughput_Mbps.
    qp_idx : int, default 0
        Which QP row of seg_bitrate_Mbps to test.
    fps_candidates : iterable[int]
        Frame-rates (fps) to evaluate.
    segment_duration_s : float, default 1.0
        Playback duration represented by each segment.
    initial_buffer_s, max_buffer_s : float
        Player buffer parameters.
    show_ratio : bool, default False
        If True plot stall **ratio** (stall_time / video_time); else raw seconds.
    **plot_kwargs : forwarded to plt.plot for styling.
    """
    stall_metric = []
    for fps in fps_candidates:
        rate_scaled = np.array([rescale_bitrate(seg_bitrate_Mbps[q], fps) for q in range(7)])

        stats = simulate_buffer(
            rate_scaled,
            net_df,
            segment_duration_s=segment_duration_s,
            initial_buffer_s=initial_buffer_s,
            max_buffer_s=max_buffer_s,
        )
        stall_s = stats["total_stall_s"][qp_idx]
        if show_ratio:
            stall_metric.append(stall_s / (len(rate_scaled) * segment_duration_s))
        else:
            stall_metric.append(stall_s)

    plt.figure(figsize=(6, 4))
    plt.plot(fps_candidates, stall_metric, marker="o", **plot_kwargs)
    plt.xlabel("Frame-rate [fps]")
    plt.ylabel("Total stall time [s]" if not show_ratio else "Stall ratio")
    plt.title(f"Stall vs fps  (QP index {qp_idx})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_stall_vs_fps_multi(seg_bitrate_Mbps,
                            net_paths,  # list[str] or list[pathlib.Path]
                            qp_idx=0,
                            fps_candidates=range(15, 91, 5),
                            segment_duration_s=1.0,
                            initial_buffer_s=2.0,
                            max_buffer_s=10.0,
                            show_ratio=False,
                            show_individual=False,
                            **plot_kwargs):
    """
    Stall-time sweep averaged over *multiple* throughput traces.

    Parameters
    ----------
    seg_bitrate_Mbps : (7, T) ndarray  (from build_fov_traces()['bitrate'])
    net_paths        : iterable of file paths
    qp_idx           : which QP row to test
    fps_candidates   : iterable of fps values
    show_ratio       : plot stall ratio instead of seconds
    show_individual  : overlay per-trace curves in light gray
    **plot_kwargs    : forwarded to plt.errorbar for styling
    """
    fps_candidates = list(fps_candidates)
    n_fps = len(fps_candidates)
    n_traces = len(net_paths)
    stall_matrix = np.zeros((n_traces, n_fps))

    for j, path in enumerate(net_paths):
        net_df = load_net_trace(path)
        for i, fps in enumerate(fps_candidates):
            rate_scaled = np.array([rescale_bitrate(seg_bitrate_Mbps[q], fps) for q in range(7)])
            stats = simulate_buffer(rate_scaled,
                                    net_df,
                                    segment_duration_s=segment_duration_s,
                                    initial_buffer_s=initial_buffer_s,
                                    max_buffer_s=max_buffer_s)
            stall_s = stats['total_stall_s'][qp_idx]
            if show_ratio:
                stall_s /= len(rate_scaled) * segment_duration_s
            stall_matrix[j, i] = stall_s

        # Optional: plot each trace’s curve
        if show_individual:
            plt.plot(fps_candidates, stall_matrix[j],
                     color="gray", alpha=0.3, linewidth=1)

    mean_stall = stall_matrix.mean(axis=0)
    std_stall = stall_matrix.std(axis=0)

    p50 = np.median(stall_matrix, axis=0)
    p05 = np.percentile(stall_matrix, 5, axis=0)
    p95 = np.percentile(stall_matrix, 95, axis=0)

    # Mean ± std error-bar plot
    plt.errorbar(fps_candidates, mean_stall, yerr=std_stall,
                 marker="o", capsize=4, color="tab:orange", label='mean')

    plt.fill_between(fps_candidates, p05, p95, color="lightblue", alpha=0.4,
                     label="5–95 percentile")
    plt.plot(fps_candidates, p50, marker="o", color="tab:blue",
             label="median")
    plt.legend()

    plt.xlabel("Frame-rate [fps]")
    plt.ylabel("Stall time [s]" if not show_ratio else "Stall ratio")
    plt.title(f"Stall vs fps (QP index {qp_idx})\n"
              f"{n_traces} network traces – mean ± 1σ")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    # plt.ylim(0, 250)
    plt.show()

    return mean_stall, std_stall  # in case you want the numbers
