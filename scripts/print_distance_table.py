import pickle

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def prepare_data(bins, file_path, only_offloaded=True):
    f = open(file_path, 'rb')
    ppg_data = pickle.load(f)
    df = pd.concat([pd.DataFrame(v) for v in ppg_data.values()])
    # df['throughput'] = (df['task_size'] + df['task_res_size']) / df['latency']
    df['throughput'] = (df['task_size'] + df['task_res_size']) / (df['tx_time'] + df['rx_time'])
    df['tx_throughput'] = df['task_size'] / df['tx_time']
    df['rx_throughput'] = df['task_res_size'] / df['rx_time']
    if only_offloaded:
        df = df[df['offloading_decision'] > 0]
        throughput_segment, ret_bins = pd.qcut(df['throughput'], q=bins, retbins=True)
        df['throughput_segment'] = throughput_segment.values.codes
    # else:
    #     df = df[df['offloading_decision'] == 0]
    # df['tx_segment'] = pd.cut(df['tx_throughput'], bins=bins, retbins=False).values.codes
    # df['size_segment'] = pd.qcut(df['task_size'], q=bins, retbins=False).values.codes
    df['size_segment'] = pd.cut(df['task_size'], bins=bins, retbins=False).values.codes
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
    data = {
        'Optimal': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/Optimal.pkl')[0],
        'CPPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/CPPG.pkl')[0],
        'PPG': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPG.pkl')[0],
        'EGreedy': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/EGreedy.pkl')[0],
        'PPO': prepare_data(bins, 'results/ppg-exp/w0_0.35_w1_0.85_w2_0.15/PPO.pkl')[0],
    }

    st = {}
    for label, df in data.items():
        stats = df.groupby(by=['quality_decision',
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
    for i, df1 in st.items():
        p = pd.DataFrame(df1).to_numpy().flatten()
        d[i] = {}
        jj = 0
        for j, df2 in st.items():
            q = pd.DataFrame(df2).to_numpy().flatten()
            if jj >= ii:
                d[i][j] = .5 * np.abs(p - q).sum()
            else:
                d[i][j] = wasserstein_distance(p, q)
            jj += 1
        ii += 1
    print(pd.DataFrame(d))
