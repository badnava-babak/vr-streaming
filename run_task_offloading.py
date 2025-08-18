from __future__ import annotations

from tqdm import tqdm

from src.chennels.two_way_channel import TwoWayChannel
from src.commons.io import load_all_videos, load_all_users, load_all_traces
from src.commons.plots import plot_metric_distribution, plot_x_vs_y
from src.commons.stats import EpisodeStats
from src.envs.elastic_task_offloading import ElasticTaskOffloadingEnv
from src.nodes.edge_server import EdgeNode
from src.nodes.vr_device import VRDevice
from src.policies.bandits_policy import BanditPolicy
from src.policies.optimal_dm import OptimalDecisionMaker

from matplotlib import pyplot as plt

import argparse
from pathlib import Path
import csv, io
import numpy as np
from src.policies.ppg_policy import MultiTaskPPGPolicy, CentralizedMultiTaskPPGPolicy, PPGPolicy


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
    p.add_argument("--verbose", type=bool, default=False, help="Whether to print")
    p.add_argument("--num_episodes", type=int, default=1000, help="Number of Episodes to train the policy")

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
    p.add_argument("--csv-log", default="results/ppg-exp",
                   help="CSV file to append results to")

    return p.parse_args()


def print_args(args):
    if args.verbose:
        """Prints all arguments and their values in a formatted way."""
        print("\n--- Program Arguments ---")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
        print("-------------------------\n")


def run_sim(args):
    N = int(args.num_users)
    C = 3

    vr_users = create_users(args)

    edge = EdgeNode(processing_rate=args.edge_proc_speed)

    multi_user_env = ElasticTaskOffloadingEnv(edge, vr_users,
                                              weights=(args.weights[0], args.weights[1], args.weights[2]))

    update_timestep = 4  # update policy every n episodes

    policy = None
    if args.policy == "Optimal":
        policy = OptimalDecisionMaker(
            action_dim=(7, C + 1),
            weights=(args.weights[0], args.weights[1], args.weights[2])  # psnr, stall time, energy
        )
        args.num_episodes = 1
    elif args.policy == "PPG":
        policy = MultiTaskPPGPolicy(num_channels=C,
                                    num_users=N,
                                    weights=(args.weights[0], args.weights[1], args.weights[2])
                                    # psnr, stall time, energy
                                    )
    elif args.policy == "CPPG":
        policy = CentralizedMultiTaskPPGPolicy(num_channels=C,
                                               num_users=N,
                                               weights=(args.weights[0], args.weights[1], args.weights[2])
                                               # psnr, stall time, energy
                                               )
    elif args.policy == "EGreedy":
        policy = BanditPolicy(num_channels=C,
                              num_users=N,
                              weights=(args.weights[0], args.weights[1], args.weights[2]))
    else:
        raise ValueError("--policy must be one of the following: Optimal, EGreedy, PPG or CPPG")

    history = []
    pbar = tqdm(range(args.num_episodes), desc="Initial Description", disable=not args.verbose)
    for ep in pbar:
        multi_user_stats = multi_user_env.run(policy=policy)
        history.append(multi_user_stats)

        multi_user_env.reset()
        if ep % update_timestep == 0:
            if isinstance(policy, PPGPolicy):
                policy.update()

        if ep % 10 == 0 and args.verbose:
            stats = multi_user_stats.summary_stats()['overall']

            pbar.set_postfix({
                'Avg. PSNR': stats['psnr_mean'],
                'Avg. Latency': stats['latency_mean'],
                'Avg. Energy Consumption': stats['energy_mean'],
                'Avg. Reward': stats['reward_mean']
            })
            # pbar.set_description(f"Episode {ep}")

    if args.verbose:
        stats = multi_user_stats.summary_stats()['overall']
        print(
            f"Episode: {ep}, Avg. PSNR: {stats['psnr_mean']:.03f}, Avg. Latency: {stats['latency_mean']:.03f}, Avg. Energy Consumption: {stats['energy_mean']:.03f}, Avg. Reward: {stats['reward_mean']:.03f}")
    return multi_user_stats


def create_users(args):
    vr_users = []
    N = int(args.num_users)
    np.random.seed(42)  # Set a seed
    ch_5g_idx = np.random.randint(0, len(channels_5g), size=(N, 2))
    ch_4g_idx = np.random.randint(0, len(channels_4g), size=(N, 2))
    ch_wigig_idx = np.random.randint(0, len(channels_wiGig), size=(N, 2))
    video_idx = np.random.randint(0, 9)
    for n in range(N):
        channels = [
            TwoWayChannel(channels_5g[ch_5g_idx[n][0]], channels_5g[ch_5g_idx[n][1]]),
            TwoWayChannel(channels_4g[ch_4g_idx[n][0]], channels_4g[ch_4g_idx[n][1]]),
            TwoWayChannel(channels_wiGig[ch_wigig_idx[n][0]], channels_wiGig[ch_wigig_idx[n][1]]),
        ]
        vr = VRDevice(channels=channels,
                      processing_rate=args.device_proc_speed,
                      cpu_freq=args.device_cpu_freq,
                      video=videos[video_idx],
                      user=video_users[video_idx][args.user_id]
                      )
        vr_users.append(vr)
    return vr_users


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    channels_5g = load_all_traces('5G')
    channels_4g = load_all_traces('4G')
    channels_wiGig = load_all_traces('WiGig')
    videos = load_all_videos()
    video_users = load_all_users()

    multi_user_stats = run_sim(args)
    overall_stats = multi_user_stats.summary_stats()['overall']

    program_args = vars(args)
    weights = {f'w{i}':w for i, w in enumerate(program_args['weights'])}

    stats_to_write = {**overall_stats, **program_args, **weights}

    log_path = Path(args.csv_log + f"/w0_{args.weights[0]}_w1_{args.weights[1]}_w2_{args.weights[2]}.csv")
    if not log_path.exists():
        with log_path.open("a", newline="") as f:
            # colum_names = 'policy,seed,num_users,video_id,user_id,device_proc_speed,device_cpu_freq,edge_proc_speed,w0,w1,w2,csv_log,'
            # colum_names += str(list(overall_stats.keys()))[1:-1].replace('\'', '').replace(' ', '')

            colum_names = list(stats_to_write.keys())
            writer = csv.DictWriter(f, fieldnames=list(colum_names))
            writer.writeheader()
            writer.writerow(stats_to_write)

            # f.write(colum_names + "\n")
            # f.write(sim_args + ',' + sim_results + "\n")

            f.close()
    else:
        with log_path.open("a", newline="") as f:
            # f.write(sim_args + ',' + sim_results + "\n")
            colum_names = list(stats_to_write.keys())
            writer = csv.DictWriter(f, fieldnames=list(colum_names))
            # writer.writeheader()
            writer.writerow(stats_to_write)
            f.close()

    overall_ = {
        'Optimal Solution: 30 Users': multi_user_stats.summary_stats()['overall'],
        #     'Optimal Solution: 1 User': single_user_stats.summary_stats()['overall'],
    }

    # plot_x_vs_y(overall_, x_label='energy', y_label='psnr')
    # plot_x_vs_y(overall_, x_label='latency', y_label='psnr')
    # plot_x_vs_y(overall_, x_label='energy', y_label='latency')
    #
    # overall_per_user_stats = {
    #     'Optimal Solution: 30 Users': multi_user_stats.summary_stats()['per_device']
    # }
    # plot_metric_distribution(overall_per_user_stats, 'energy_mean')
