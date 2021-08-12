import pandas as pd
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def analyze_all(df, df_score):
    print('ANALYZE ALL')
    df_score = df_score.set_index('index')
    df_score = df_score.reindex(index=df['index'])
    df_score = df_score.reset_index()
    m1 = df_score['m1'].tolist()
    m2 = df_score['m2'].tolist()
    m1 = np.array(m1)
    m2 = np.array(m2)
    m1_corr = stats.spearmanr(m1, df['conf_ep'].to_numpy())
    m2_corr = stats.spearmanr(m2, df['conf_ep'].to_numpy())
    print('correlation between m1 and confidence', m1_corr)
    print('correlation between m2 and confidence', m2_corr)
    return m1_corr[0], m2_corr[0]
    # df_.to_csv(os.path.join(analysis_dir, f'{file_[:-4]}_OVERLAP_SCORE.csv'))

def split_category(df, col_to_split, split_value):
    if split_value:
        return df.groupby(df[col_to_split] > split_value)
    return df.groupby(df[col_to_split])

def analyze_by_category(df, df_score, col_to_split, split_value=None):
    print('ANALYZE BY VALUE')
    m1_corrs = {}
    m2_corrs = {}
    cols = np.unique(df[col_to_split])
    for col in cols:
        df_ = df.loc[df[col_to_split]==col]
        df_score_ep = df_score.loc[df_score['index'].isin(df_['index'].tolist())]
        df_= df_.sort_values(by=['index'])
        df_score_ep=df_score_ep.sort_values(by=['index'])
        m1 = df_score_ep['m1'].tolist()
        m2 = df_score_ep['m2'].tolist()
        m1 = np.array(m1)
        m2 = np.array(m2)

        column_value = df_[column_to_split].tolist()[0]
        print(f'{column_to_split} VALUE = {column_value}')
        m1_corr = stats.spearmanr(m1, df_['conf_ep'].to_numpy())
        m2_corr = stats.spearmanr(m2, df_['conf_ep'].to_numpy())
        print('correlation between m1 and confidence', m1_corr)
        print('correlation between m2 and confidence', m2_corr)
        m1_corrs[column_value] = m1_corr[0]
        m2_corrs[column_value] = m2_corr[0]
    return m1_corrs, m2_corrs

def plot_trend(plot_dir, mode, y, type):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(y)
    fig.savefig(os.path.join(plot_dir, f'{mode}_{type}_TREND.png'))

if __name__ == '__main__':
    column_to_split = 'pred_label'
    mode = 'OOD'
    file_with_score = f'/home/jusun/adila001/robust_nlp/cartography/model_output_RTE_4/ANALYSIS_CLEAN/OVERLAP_SCORE/{mode}_SCORE.csv'
    df_score = pd.read_csv(file_with_score)
    analysis_dir = '/home/jusun/adila001/robust_nlp/cartography/model_output_RTE_4/ANALYSIS_CLEAN/SORTED'
    ep_num = 99
    m1 = []
    m2 = []
    col_vals = np.unique(df_score[column_to_split].tolist())

    m1_cat = {}
    m2_cat = {}
    for val in col_vals:
        m1_cat[val] =[]
        m2_cat[val] = []

    for i in range(ep_num+1):
        print(f'EPOCH = {i}')
        df = pd.read_csv(f'{analysis_dir}/{mode}_CONF_ep_{i}_SORTED.csv')
        m1_ep, m2_ep = analyze_all(df, df_score)
        m1.append(m1_ep)
        m2.append(m2_ep)
        m1_cat_ep, m2_cat_ep = analyze_by_category(df, df_score, column_to_split)
        for val in col_vals:
            if val in m1_cat_ep.keys():
                m1_cat[val].append(m1_cat_ep[val])
                # np.insert(m1_cat[val], i, m1_cat_ep[val])
            else:
                m1_cat[val].append(0)
            if val in m2_cat_ep.keys():
                m2_cat[val].append(m2_cat_ep[val])
                # np.insert(m2_cat[val], i, m2_cat_ep[val])
            else:
                m1_cat[val].append(0)
    plot_dir = os.path.join(f'{"/".join(analysis_dir.split("/")[:-1])}','PLOT')
    plot_trend(plot_dir, mode, m1, 'M1')
    plot_trend(plot_dir, mode, m2, 'M2')
    for key in m1_cat.keys():
        plot_trend(plot_dir, mode, m1_cat[key], f'M1_{key}')
    for key in m2_cat.keys():
        plot_trend(plot_dir, mode, m2_cat[key], f'M2_{key}')
    ep_arr = np.linspace(0, ep_num, ep_num+1, True, int)[0]
    print('\n')

    m1 = np.abs(m1)
    m2 = np.abs(m2)

    print('m1 ALL trend', stats.kendalltau(m1, ep_arr))
    print('m2 ALL trend', stats.kendalltau(m2, ep_arr))
    print('\n')
    #
    for val in col_vals:
        m1_cat[val] = np.abs(m1_cat[val])
        m2_cat[val] = np.abs(m2_cat[val])

        m1_cat[val] = np.asarray(m1_cat[val])
        m1_cat_val = m1_cat[val][~np.isnan(m1_cat[val])]
        m1_ep_arr = np.linspace(0, len(m1_cat_val), len(m1_cat_val),  False, int)[0]
        # print(m1_ep_arr)
        print(f'm1 {val} trend', stats.kendalltau(m1_cat_val, m1_ep_arr))

        m2_cat[val] = np.asarray(m2_cat[val])
        m2_cat_val = m2_cat[val][~np.isnan(m2_cat[val])]
        m2_ep_arr = np.linspace(0, len(m2_cat_val), len(m2_cat_val),  False, int)[0]
        print(f'm2 {val} trend', stats.kendalltau(m2_cat_val, m2_ep_arr))

    print('\n')
    print('M1', m1)
    print('M2', m2)

    print('M1 CAT', m1_cat)
    print('M2 CAT', m2_cat)






