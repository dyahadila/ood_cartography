import pandas as pd
import os
import re
from scipy import stats
import spacy
from tqdm import tqdm

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
    print('ANALYZE ALL')
    m1 = df['m1']
    m2 = df['m2']
    print('SPEARMAN')
    print('correlation between m1 and variance', stats.spearmanr(m1, df_['var_ep'].to_numpy()))
    print('correlation between m2 and variance', stats.spearmanr(m2, df_['var_ep'].to_numpy()))
    df_.to_csv(os.path.join(analysis_dir, f'{file_[:-4]}_OVERLAP_SCORE.csv'))

def split_category(df, col_to_split, split_value):
    if split_value:
        return df.groupby(df[col_to_split] > split_value)
    return df.groupby(df[col_to_split])

def analyze_by_category(df, col_to_split, split_value=None):
    print('ANALYZE BY VALUE')
    for file_idx, df_ in enumerate([x for _, x in split_category(df, col_to_split, split_value)]):
        print(f'{column_to_split} VALUE = {df_[column_to_split].tolist()[0]}')
        m1 = df_['m1']
        m2 = df_['m2']
        print('SPEARMAN')
        print('correlation between m1 and confidence', stats.spearmanr(m1, df_['conf_ep'].to_numpy()))
        print('correlation between m2 and confidence', stats.spearmanr(m2, df_['conf_ep'].to_numpy()))
        print('\n')

        # df_.to_csv(os.path.join(analysis_dir, f'{file_[:-4]}_OVERLAP_SCORE_BY_{column_to_split}_{file_idx}.csv'))

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    column_to_split = 'conf_ep'
    analysis_dir = '/home/jusun/adila001/robust_nlp/cartography/model_output_RTE/overlap_words_analysis/ALL'
    file_ = 'OOD_LEXICAL.csv'
    df = pd.read_csv(os.path.join(analysis_dir,file_))
    # df = df.loc[df['pred_label']=='not_entailment']
    analyze_all(df)
    # split_criteria = {
    #     "confidence":
    # }
    # analyze_by_category(df, column_to_split, 0.6)




