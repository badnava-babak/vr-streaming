from __future__ import annotations

from src.chennels.two_way_channel import TwoWayChannel
from src.commons.io import load_all_videos, load_all_users, load_all_traces
from src.commons.plots import plot_energy_vs_latency, plot_energy_vs_psnr
from src.commons.stats import EpisodeStats
from src.envs.elastic_task_offloading import ElasticTaskOffloadingEnv
from src.nodes.edge_server import EdgeNode
from src.nodes.vr_device import VRDevice
from src.policies.optimal_dm import OptimalDecisionMaker

from matplotlib import pyplot as plt

import argparse
from pathlib import Path
import csv, io

def args_to_csv_row(args, *, value_sep=","):
    buf = io.StringIO()
    csv.writer(buf).writerow(
        [
            value_sep.join(map(str, v)) if isinstance(v, (list, tuple))
            else v
            for v in vars(args).values()
        ]
    )
    return buf.getvalue().strip("\r\n").replace("\"", '')


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a simulation or post-process an EpisodeStats pickle "
                    "and log the metrics to a CSV file."
    )
    p.add_argument("--policy", required=False, default='Optimal',
                   help="Policy / algorithm name (string)")

    p.add_argument("--seed", type=int, default=42, help="RNG seed or run id")

    p.add_argument("--num-users", type=int, default=5, help="Number of Users in the Environments")

    p.add_argument("--video-id", type=int, default=0, help="Video ID")
    p.add_argument("--user-id", type=int, default=0, help="User ID for Head Navigation Data")
    p.add_argument("--device-proc-speed", type=float, default=200.e6,
                   help="Edge Device Processor Speed in bps")
    p.add_argument("--device-cpu-freq", type=float, default=2.4e9,
                   help="Edge Device CPU Frequency")

    p.add_argument("--edge-proc-speed", type=float, default=6.e9,
                   help="Edge Server Processor Speed in bps")

    p.add_argument("--weights", type=float, nargs=3,
                   metavar=("w0", "w1", "w2"),
                   default=(1.0, 1.0, 1.0),
                   help="Weights of different objective functions. PSNR, Stall Time, and Energy Consumption")

    # p.add_argument("--stats-file", required=True, help="Path to pickled EpisodeStats object")
    p.add_argument("--csv-log", default="results/metrics.csv",
                   help="CSV file to append results to")

    return p.parse_args()


def run_sim(args):
    channels = [
        TwoWayChannel(channels_5g[0], channels_5g[1]),
        TwoWayChannel(channels_4g[2], channels_4g[3]),
        TwoWayChannel(channels_wiGig[0], channels_wiGig[1]),
    ]
    edge = EdgeNode(processing_rate=args.edge_proc_speed)
    vr = VRDevice(channels=channels,
                  processing_rate=args.device_proc_speed,
                  cpu_freq=args.device_cpu_freq,
                  video=videos[args.video_id],
                  user=video_users[args.video_id][args.user_id]
                  )
    optimal_policy = OptimalDecisionMaker(
        action_dim=(7, len(channels) + 1),
        weights=(args.weights[0], args.weights[1], args.weights[2])  # psnr, stall time, energy
    )
    multi_user_env = ElasticTaskOffloadingEnv(edge, [vr for _ in range(args.num_users)])
    multi_user_stats = multi_user_env.run(policy=optimal_policy)

    return multi_user_stats


if __name__ == "__main__":
    args = parse_args()

    channels_5g = load_all_traces('5G')
    channels_4g = load_all_traces('4G')
    channels_wiGig = load_all_traces('WiGig')
    videos = load_all_videos()
    video_users = load_all_users()

    multi_user_stats = run_sim(args)
    overall_stats = multi_user_stats.summary_stats()['overall']
    sim_args = args_to_csv_row(args)
    sim_results = str(list(overall_stats.values()))[1:-1]

    log_path = Path(args.csv_log)
    if not log_path.exists():
        with log_path.open("a", newline="") as f:
            colum_names = 'policy,seed,num_users,video_id,user_id,device_proc_speed,device_cpu_freq,edge_proc_speed,w0,w1,w2,csv_log'
            colum_names += str(list(overall_stats.keys()))[1:-1].replace('\'', '')
            # writer = csv.DictWriter(f, fieldnames=list(colum_names))
            # writer.writeheader()
            # writer.writerow(colum_names)
            f.write(colum_names + "\n")
            f.write(sim_args + ',' + sim_results + "\n")
    else:
        with log_path.open("a", newline="") as f:
            f.write(sim_args + ',' + sim_results + "\n")

    # overall_ = {
    #     'Optimal Solution: 30 Users': multi_user_stats.summary_stats()['overall'],
    #     'Optimal Solution: 1 User': single_user_stats.summary_stats()['overall'],
    # }
    # plot_energy_vs_psnr(overall_)
    # plot_energy_vs_latency(overall_)
