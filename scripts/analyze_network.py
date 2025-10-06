import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.print_distance_table import prepare_data, load_raw_data
from src.commons.plots import get_label
import pickle
def load_raw_data(file_path):
    f = open(file_path, 'rb')
    ppg_data = pickle.load(f)
    df = pd.concat([pd.DataFrame(v) for v in ppg_data.values()])
    df['p_psnr'] = df['psnr'].copy()
    df.loc[df['latency'] > 1, 'p_psnr'] = 0
    return df

if __name__ == '__main__':
    file_path = 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl'
    bins = 3

    u = '8u'

    exp_name = 'multi-video'

    data_all = {
        'Optimal': load_raw_data(f'results/{exp_name}/w0_0.35_w1_0.85_w2_0.15/{u}/Optimal.pkl'),
        'PPG': load_raw_data(f'results/{exp_name}/w0_0.35_w1_0.85_w2_0.15/{u}/PPG.pkl'),
        'CPPG': load_raw_data(f'results/{exp_name}/w0_0.35_w1_0.85_w2_0.15/{u}/CPPG.pkl')
    }

    df = data_all['PPG']
    df = df[df['offloading_decision'] > 0]
    df['throughput'] = (df['task_size'] + df['task_res_size']) / (df['tx_time'] + df['rx_time'])
    throughput_segment, ret_bins = pd.cut(df['throughput'], bins=bins, retbins=True)
    df['throughput_segment'] = throughput_segment.values.codes
    # df = df[~df['video_id'].isin([1, 8])]
    stats = df.groupby(by='throughput_segment').agg(['mean', 'std']).sort_values(
        by='throughput_segment')
    for index, row in stats.iterrows():
        # row_str = f"${row[('uplink_wigig_rates', 'mean')] / 1e9:.2f} \pm {row[('uplink_wigig_rates', 'std')] / 1e9:.2f}$ & "
        row_str = f"${row[('throughput', 'mean')] / 1e9:.2f} \pm {row[('throughput', 'std')] / 1e9:.2f}$ & "
        # row_str += f"${row[('downlink_wigig_rates', 'mean')] / 1e9:.2f} \pm {row[('downlink_wigig_rates', 'std')] / 1e9:.2f}$ & "
        # row_str += f"${row[('rewards', 'mean')]:.2f}\pm {row[('rewards', 'std')]:.2f}$ &"
        row_str += f"${row[('psnr', 'mean')]:.2f} \pm {row[('psnr', 'std')]:.2f}$ & "
        row_str += f"${row[('latency', 'mean')]:.2f} \pm {row[('latency', 'std')]:.2f}$ & "
        row_str += f"${row[('energy_consumption', 'mean')]:.2f} \pm {row[('energy_consumption', 'std')]:.2f}$ \\\\"
        print(index, row_str)