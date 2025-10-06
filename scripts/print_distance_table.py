import pickle

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

def load_raw_data(file_path):
    f = open(file_path, 'rb')
    ppg_data = pickle.load(f)
    return ppg_data

def prepare_data(bins, file_path, only_offloaded=True):
    f = open(file_path, 'rb')
    ppg_data = pickle.load(f)
    df = pd.concat([pd.DataFrame(v) for v in ppg_data.values()])
    # df['task_size'] = df['task_size'] / 1e9
    # df['task_res_size'] = df['task_res_size'] / 1e9
    # df['throughput'] = (df['task_size'] + df['task_res_size']) / df['latency']
    df['throughput'] = (df['task_size'] + df['task_res_size']) / (df['tx_time'] + df['rx_time'])
    # df['throughput'] = df['throughput'] / 1e9
    df['tx_throughput'] = df['task_size'] / df['tx_time']
    df['rx_throughput'] = df['task_res_size'] / df['rx_time']
    if only_offloaded:
        df = df[df['offloading_decision'] > 0]
        throughput_segment, ret_bins = pd.cut(df['throughput'], bins=bins, retbins=True)
        df['throughput_segment'] = throughput_segment.values.codes
    # else:
    #     df = df[df['offloading_decision'] == 0]
    # df['tx_segment'] = pd.cut(df['tx_throughput'], bins=bins, retbins=False).values.codes
    # df['size_segment'] = pd.qcut(df['task_size'], q=bins, retbins=False).values.codes
    df['size_segment'] = pd.cut(df['task_size'], bins=bins, retbins=False).values.codes
    df['up_5g_segment'] = pd.cut(df['uplink_5g_rates'], bins=np.array([-1.77e+06,  5.90e+08,  1.18e+09,  1.77e+09])).values.codes
    # df['rx_segment'] = pd.cut(df['rx_throughput'], bins=bins, retbins=False).values.codes

    # df['throughput_segment'] = pd.qcut(df['throughput'], q=bins, retbins=False).values.codes
    df['offloaded'] = (df['offloading_decision'] > 0).astype(int)
    df['5g'] = (df['offloading_decision'] == 1).astype(int)
    df['4g'] = (df['offloading_decision'] == 2).astype(int)
    df['wigig'] = (df['offloading_decision'] == 3).astype(int)
    for i in range(7):
        df[f'q_{i}'] = (df['quality_decision'] == i).astype(int)

    return df, bins


if __name__ == '__main__':
    file_path = 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl'
    bins = 3

    # opt, _ = prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl')
    u = '8u'
    exp='multi-video'
    # exp='video-2'

    data_raw = {
        'Optimal': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{exp}/{u}/Optimal.pkl'),
        'CPPG': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{exp}/{u}/CPPG.pkl'),
        'IPPG': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{exp}/{u}/PPG.pkl'),
        'EGreedy': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{exp}/{u}/EGreedy.pkl'),
        'PPO': load_raw_data(f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{exp}/{u}/PPO.pkl'),
    }


    # data = {
    #     'Optimal': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/Optimal.pkl')[0],
    #     'CPPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/CPPG.pkl')[0],
    #     'IPPG': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPG.pkl')[0],
    #     'EGreedy': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/EGreedy.pkl')[0],
    #     'PPO': prepare_data(bins, f'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/{u}/PPO.pkl')[0],
    # }

    st = {}
    for label in data_raw.keys():
        dff = pd.concat([pd.DataFrame(v) for v in data_raw[label].values()])
        stats = dff.groupby(by=['quality_decision',
                               'offloading_decision']
                           ).agg(count=('latency', 'count'))
        st[label] = {}
        for i in range(7):
            st[label][i] = {}
            for j in range(4):
                st[label][i][j] = 0.
                if (i, j) in stats.index.to_list():
                    # st += f"{i} : {stats['latency'][i] / stats['latency'].sum():.2f}, "
                    st[label][i][j] = stats['count'].loc[(i, j)] / stats['count'].sum()
                else:
                    st[label][i][j] = 0.

    d = {}
    ii = 0
    for i in st.keys():
        p = pd.DataFrame(st[i]).to_numpy().flatten()
        d[i] = {}
        jj = 0
        for j in st.keys():
            q = pd.DataFrame(st[j]).to_numpy().flatten()
            if jj >= ii:
                d[i][j] = .5 * np.abs(p - q).sum()
            else:
                d[i][j] = wasserstein_distance(range(q.shape[0]), range(p.shape[0]), q, p)
            jj += 1
        ii += 1
    print(pd.DataFrame(d))
