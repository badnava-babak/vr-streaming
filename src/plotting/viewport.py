from typing import Dict, Sequence

import numpy as np


SEG_SIZE = 30  # number of frames per aggregation bucket


def fov_to_tiles(yaw, pitch,
                 hfov=110, vfov=110,
                 tiles_x=8, tiles_y=8):
    """
    Returns a (tiles_y, tiles_x) boolean mask.
    ERP is assumed to cover yaw ∈ [-180°, 180°], pitch ∈ [-90°, 90°].
    """
    # 1. Normalise yaw, pitch to [-180,180], [-90,90]
    yaw = ((yaw + 180) % 360) - 180
    pitch = np.clip(pitch, -90, 90)

    # 2. Compute FoV bounds
    yaw_min = yaw - hfov / 2;
    yaw_max = yaw + hfov / 2
    pitch_min = pitch - vfov / 2;
    pitch_max = pitch + vfov / 2

    # 3. Build tile grid
    ys = np.linspace(-90, 90, tiles_y + 1)
    xs = np.linspace(-180, 180, tiles_x + 1)

    mask = np.zeros((tiles_y, tiles_x), bool)
    for i in range(tiles_y):
        for j in range(tiles_x):
            if (pitch_min < ys[i + 1]) and (pitch_max > ys[i]) and \
                    (yaw_min < xs[j + 1]) and (yaw_max > xs[j]):
                mask[i, j] = True
    return mask.ravel()  # length-64


def _aggregate_segments(arr: np.ndarray, seg_size: int = SEG_SIZE, op="mean") -> np.ndarray:
    """Aggregate *arr* (shape 7 × N) into 7 × T using *op* over segments."""
    q, N = arr.shape
    num_seg = int(np.ceil(N / seg_size))
    out = np.zeros((q, num_seg))
    for s in range(num_seg):
        sl = slice(s * seg_size, min((s + 1) * seg_size, N))
        if op == "sum":
            out[:, s] = arr[:, sl].sum(axis=1)
        else:  # mean
            out[:, s] = arr[:, sl].mean(axis=1)
    return out


def build_fov_traces(
        bitrate_video: np.ndarray,
        ymse_video: np.ndarray,
        pose: np.ndarray,
        bitdepth: int,
        *,
        hfov: float = 110,
        vfov: float = 110,
        tiles_x: int = 8,
        tiles_y: int = 8,
        force_deg: bool = None,
) -> Dict[str, np.ndarray]:
    """Return FoV‑weighted traces **per 30‑frame segment**.

    Output arrays shapes: (7, T_seg) where T_seg = ceil(N / 30).
    Bitrate is averaged over the segment (Mbit/s), YMSE & PSNR are averaged.
    """
    N = pose.shape[0]
    traces_br = np.zeros((7, N))
    traces_ym = np.zeros((7, N))
    traces_ps = np.zeros((7, N))
    max_i2 = (2 ** bitdepth - 1) ** 2

    for t in range(N):
        yaw, pitch = pose[t, :2]
        if force_deg is None:
            deg = not (abs(yaw) <= 2 * np.pi and abs(pitch) <= np.pi)
        else:
            deg = force_deg
        if not deg:
            yaw, pitch = np.rad2deg([yaw, pitch])
        mask = fov_to_tiles(yaw, pitch, hfov=hfov, vfov=vfov, tiles_x=tiles_x, tiles_y=tiles_y)
        for q in range(7):
            traces_br[q, t] = bitrate_video[q, mask, t].sum()
            ymse_fov = ymse_video[q, mask, t].mean()
            traces_ym[q, t] = ymse_fov
            traces_ps[q, t] = 10 * np.log10(max_i2 / ymse_fov)

    # Aggregate every 30 frames (≈1s)
    return {
        "bitrate": _aggregate_segments(traces_br, SEG_SIZE, op="sum"),
        "ymse": _aggregate_segments(traces_ym, SEG_SIZE, op="mean"),
        "psnr": _aggregate_segments(traces_ps, SEG_SIZE, op="mean"),
    }


# -----------------------------------------------------------------------------
# 4. DATASET‑LEVEL AGGREGATION (30‑frame buckets)
# -----------------------------------------------------------------------------

def aggregate_traces(
        rd: Dict[str, Sequence[np.ndarray]],
        hmd_data,
        *,
        user_idx: int = 0,
        hfov: float = 110,
        vfov: float = 110,
        tiles_x: int = 8,
        tiles_y: int = 8,
        force_deg: bool = None,
) -> Dict[str, np.ndarray]:
    """Aggregate **mean** traces across all videos, bucketed per 30 frames."""
    bitrate_list = rd["bitrate"]
    ymse_list = rd["ymse"]
    bitdepths = rd["bitdepth"]
    num_vids = len(bitrate_list)

    # Determine global maximum number of segments
    max_seg = 0
    segs_per_video = []
    for br in bitrate_list:
        N = br.shape[2]
        nseg = int(np.ceil(N / SEG_SIZE))
        segs_per_video.append(nseg)
        max_seg = max(max_seg, nseg)

    acc_br = np.full((7, max_seg), np.nan)
    acc_ym = np.full((7, max_seg), np.nan)
    acc_ps = np.full((7, max_seg), np.nan)

    for v in range(num_vids):
        traces = build_fov_traces(
            bitrate_list[v],
            ymse_list[v],
            hmd_data[v, user_idx],
            bitdepths[v],
            hfov=hfov,
            vfov=vfov,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            force_deg=force_deg,
        )
        segs = segs_per_video[v]
        sl = slice(0, segs)
        # mean across videos (ignoring NaNs)
        if np.isnan(acc_br[0, sl]).all():
            acc_br[:, sl] = traces["bitrate"]
            acc_ym[:, sl] = traces["ymse"]
            acc_ps[:, sl] = traces["psnr"]
        else:
            acc_br[:, sl] = np.nanmean(
                np.stack([acc_br[:, sl], traces["bitrate"]]), axis=0
            )
            acc_ym[:, sl] = np.nanmean(
                np.stack([acc_ym[:, sl], traces["ymse"]]), axis=0
            )
            acc_ps[:, sl] = np.nanmean(
                np.stack([acc_ps[:, sl], traces["psnr"]]), axis=0
            )

    return {"bitrate": acc_br, "ymse": acc_ym, "psnr": acc_ps}


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEGMENT DOWNLOAD‑TIME SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_tx_times(
        seg_bitrate_Mbps: np.ndarray,  # (T,)
        net_df,
        *,
        seg_duration_s: float = SEG_SIZE / 30.0,  # default 1s
) -> np.ndarray:
    """Return an array of simulated **download times per segment**.

    The network trace *net_df* must contain `time_ms` and `throughput_Mbps`.
    Unused residual capacity in a measurement interval carries over to the next
    segment if a download finishes mid‑interval.  If the trace ends before all
    segments are downloaded, remaining segments are set to *np.inf*.
    """
    size_bits = seg_bitrate_Mbps  * seg_duration_s # (T,) bits
    T = len(size_bits)
    tx_times = np.full(T, np.inf)

    times = net_df["time_ms"].values.astype(float) / 1000.0
    thr_bits = net_df["throughput_Mbps"].values.astype(float) * 1e6

    idx = 0
    current_time = times[0]

    for s in range(T):
        bits_left = size_bits[s]
        elapsed = 0.0
        while bits_left > 0 and idx < len(times) - 1:
            t_end = times[idx + 1]
            dt = t_end - current_time
            cap = thr_bits[idx] * dt
            if cap >= bits_left:
                dt_need = bits_left / thr_bits[idx]
                elapsed += dt_need
                current_time += dt_need  # remains in same interval
                bits_left = 0
            else:
                elapsed += dt
                bits_left -= cap
                idx += 1
                current_time = times[idx]
        if bits_left == 0:
            tx_times[s] = elapsed
        else:
            break  # network trace exhausted
    return tx_times


