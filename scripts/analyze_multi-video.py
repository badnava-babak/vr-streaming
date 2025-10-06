import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.print_distance_table import prepare_data
from src.commons.plots import get_label
import pickle


def load_raw_data(file_path):
    f = open(file_path, 'rb')
    ppg_data = pickle.load(f)
    df = pd.concat([pd.DataFrame(v) for v in ppg_data.values()])
    df['p_psnr'] = df['psnr'].copy()
    df['deadline_violation'] = (df['latency'] > 1.).astype(int).copy()
    df.loc[df['latency'] > 1, 'p_psnr'] = 0
    return df


if __name__ == '__main__':
    file_path = 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl'
    bins = 3

    # opt, _ = prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl')
    u = '8u'

    exp_name = 'multi-video'

    data_all = {
        'Optimal': load_raw_data(f'results/{exp_name}/w0_0.35_w1_0.85_w2_0.15/{u}/Optimal.pkl'),
        'PPG': load_raw_data(f'results/{exp_name}/w0_0.35_w1_0.85_w2_0.15/{u}/PPG.pkl'),
        'CPPG': load_raw_data(f'results/{exp_name}/w0_0.35_w1_0.85_w2_0.15/{u}/CPPG.pkl')
    }

    video_names = ['Academic', 'Basketball', 'Bridge', 'GateNight',
                   'Runner', 'SiyuanGate', 'SouthGate', 'StudyRoom', 'Sward']

    df = data_all['CPPG']
    ippg_stats = data_all['PPG'].groupby(by='video_id').agg(['mean', 'std']).sort_values(by='video_id')
    cppg_stats = data_all['CPPG'].groupby(by='video_id').agg(['mean', 'std']).sort_values(by='video_id')
    optimal_stats = data_all['Optimal'].groupby(by='video_id').agg(['mean', 'std']).sort_values(by='video_id')

    for index in range(8):
        row_ippg = ippg_stats.iloc[index]
        row_cppg = cppg_stats.iloc[index]
        row_optimal = optimal_stats.iloc[index]

        row_str = f"{video_names[index]} &"

        row_str += f"${row_ippg[('rewards', 'mean')]:.2f}$ &"
        row_str += f"${row_ippg[('psnr', 'mean')]:.2f} \pm {row_ippg[('psnr', 'std')]:.2f}$ &"
        row_str += f"${row_ippg[('p_psnr', 'mean')]:.2f} \pm {row_ippg[('p_psnr', 'std')]:.2f}$ &"
        row_str += f"${row_ippg[('latency', 'mean')]:.2f} \pm {row_ippg[('latency', 'std')]:.2f}$ &"
        row_str += f"${row_ippg[('energy_consumption', 'mean')]:.2f} \pm {row_ippg[('energy_consumption', 'std')]:.2f}$ &"
        row_str += f"${row_ippg[('deadline_violation', 'mean')]*100:.2f}$ &"

        row_str += f"${row_cppg[('rewards', 'mean')]:.2f}$ &"
        row_str += f"${row_cppg[('psnr', 'mean')]:.2f} \pm {row_cppg[('psnr', 'std')]:.2f}$ &"
        row_str += f"${row_cppg[('p_psnr', 'mean')]:.2f} \pm {row_cppg[('p_psnr', 'std')]:.2f}$ &"
        row_str += f"${row_cppg[('latency', 'mean')]:.2f} \pm {row_cppg[('latency', 'std')]:.2f}$ &"
        row_str += f"${row_cppg[('energy_consumption', 'mean')]:.2f} \pm {row_cppg[('energy_consumption', 'std')]:.2f}$ &"
        row_str += f"${row_cppg[('deadline_violation', 'mean')] * 100:.2f}$ "

        # row_str += f"${row_optimal[('rewards', 'mean')]:.2f}$ &"
        # row_str += f"${row_optimal[('p_psnr', 'mean')]:.2f} \pm {row_optimal[('p_psnr', 'std')]:.2f}$ &"
        # row_str += f"${row_optimal[('latency', 'mean')]:.2f} \pm {row_optimal[('latency', 'std')]:.2f}$ &"
        # row_str += f"${row_optimal[('energy_consumption', 'mean')]:.2f} \pm {row_optimal[('energy_consumption', 'std')]:.2f}$ &"
        # row_str += f"${row_optimal[('deadline_violation', 'mean')]:.2f}$"

        row_str += "\\\\"
        print(row_str)

    ippg_all = data_all['PPG'].agg(['mean', 'std'])
    cppg_all = data_all['CPPG'].agg(['mean', 'std'])
    optimal_all = data_all['Optimal'].agg(['mean', 'std'])

    print('\n\n')
    # row_str = f"Total & ${stats_all.loc['mean']['rewards']:.2f}$ & "
    row_str = "Average &"
    row_str += f"${ippg_all.loc['mean']['rewards']:.2f}$ &"
    row_str += f"${ippg_all.loc['mean']['psnr']:.2f} \pm {ippg_all.loc['std']['psnr']:.2f}$ &"
    row_str += f"${ippg_all.loc['mean']['p_psnr']:.2f} \pm {ippg_all.loc['std']['p_psnr']:.2f}$ &"
    row_str += f"${ippg_all.loc['mean']['latency']:.2f} \pm {ippg_all.loc['std']['latency']:.2f}$ &"
    row_str += f"${ippg_all.loc['mean']['energy_consumption']:.2f} \pm {ippg_all.loc['std']['energy_consumption']:.2f}$ &"
    row_str += f"${ippg_all.loc['mean']['deadline_violation']:.2f}$ &"

    row_str += f"${cppg_all.loc['mean']['rewards']:.2f} $ &"
    row_str += f"${cppg_all.loc['mean']['psnr']:.2f} \pm {cppg_all.loc['std']['psnr']:.2f}$ &"
    row_str += f"${cppg_all.loc['mean']['p_psnr']:.2f} \pm {cppg_all.loc['std']['p_psnr']:.2f}$ &"
    row_str += f"${cppg_all.loc['mean']['latency']:.2f} \pm {cppg_all.loc['std']['latency']:.2f}$ &"
    row_str += f"${cppg_all.loc['mean']['energy_consumption']:.2f} \pm {cppg_all.loc['std']['energy_consumption']:.2f}$ &"
    row_str += f"${cppg_all.loc['mean']['deadline_violation']:.2f}$ "

    # row_str += f"${optimal_all.loc['mean']['p_psnr']:.2f} \pm {optimal_all.loc['std']['p_psnr']:.2f}$ &"
    # row_str += f"${optimal_all.loc['mean']['latency']:.2f} \pm {optimal_all.loc['std']['latency']:.2f}$ &"
    # row_str += f"${optimal_all.loc['mean']['energy_consumption']:.2f} \pm {optimal_all.loc['std']['energy_consumption']:.2f}$ &"
    # row_str += f"${optimal_all.loc['mean']['deadline_violation']:.2f}$ \\\\"
    row_str += "\\\\"
    print(row_str)
