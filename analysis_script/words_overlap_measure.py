import pandas as pd
import os
import re
from scipy import stats
import spacy
from tqdm import tqdm


nlp = spacy.load("en_core_web_sm")
column_to_split = 'pred_label'
mode = 'OOD'
analysis_dir = '/home/jusun/adila001/robust_nlp/cartography/model_output_RTE_4/ANALYSIS_CLEAN/SORTED'
result_dir = '/home/jusun/adila001/robust_nlp/cartography/model_output_RTE_4/ANALYSIS_CLEAN/OVERLAP_SCORE'

def tokenize(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()


    s1 = s1.split(" ")
    s2 = s2.split(" ")

    s1 = [re.sub(r'\W+', '', s1_) for s1_ in s1 if s1_ != '']
    s2 = [re.sub(r'\W+', '', s2_) for s2_ in s2 if s2_ != '']

    return s1, s2

def count_overlap(t1, t2):
    return len(set(t1).intersection(set(t2)))

def analyze_all(df_):
    print('ANALYZING ALL')
    m1 = []
    m2 = []
    for i, row in tqdm(df_.iterrows()):
        s1 = row['sentence1']
        s2 = row['sentence2']
        try:
            t1 = nlp(s1)
            t2 = nlp(s2)
        except:
            df_.drop([i])
        t1 = [str(t_) for t_ in t1]
        t2 = [str(t_) for t_ in t2]
        overlap_count = count_overlap(t1, t2)
        m1.append(overlap_count / len(t1))
        m2.append(overlap_count / len(t2))

    df_['m1'] = m1
    df_['m2'] = m2

    print('SPEARMAN')
    print('correlation between m1 and confidence', stats.spearmanr(m1, df_['conf_ep'].to_numpy()))
    print('correlation between m2 and confidence', stats.spearmanr(m2, df_['conf_ep'].to_numpy()))
    df_.to_csv(os.path.join(result_dir, f'{mode}_SCORE.csv'))

def split_category(df, col_to_split, split_value):
    if split_value:
        return df.groupby(df[col_to_split] > split_value)
    return df.groupby(df[col_to_split])

def analyze_by_category(df, col_to_split, split_value=None):
    print('ANALYZE BY VALUE')
    for file_idx, df_ in enumerate([x for _, x in split_category(df, col_to_split, split_value)]):
        print(f'{column_to_split} VALUE = {df_[column_to_split].tolist()[0]}')
        m1 = []
        m2 = []
        for i,row in tqdm(df_.iterrows()):
            s1 = row['sentence1']
            s2 = row['sentence2']
            try:
                t1 = nlp(s1)
                t2 = nlp(s2)
            except:
                df_.drop([i])
            t1 = [str(t_) for t_ in t1]
            t2 = [str(t_) for t_ in t2]
            overlap_count = count_overlap(t1, t2)
            m1.append(overlap_count/len(t1))
            m2.append(overlap_count / len(t2))

        df_['m1'] = m1
        df_['m2'] = m2
        print('SPEARMAN')
        print('correlation between m1 and confidence', stats.spearmanr(m1, df_['conf_ep'].to_numpy()))
        print('correlation between m2 and confidence', stats.spearmanr(m2, df_['conf_ep'].to_numpy()))
        print('\n')

        # df_.to_csv(os.path.join(analysis_dir, f'{file_[:-4]}_OVERLAP_SCORE_BY_{column_to_split}_{file_idx}.csv'))

if __name__ == '__main__':
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    ep_num = 99
    file_ = f'{mode}_CONF_ep_{ep_num}_SORTED.csv'
    df = pd.read_csv(os.path.join(analysis_dir,file_))
    # df = df.loc[df['pred_label']=='not_entailment']
    analyze_all(df)
    # split_criteria = {
    #     "confidence":
    # }
    analyze_by_category(df, column_to_split)




