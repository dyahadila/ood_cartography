import pandas as pd
import os
import re
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_sm")

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

def Diff(li1, li2):
    return len(list(set(li1) - set(li2)))

def get_similar_subsentence(s1, s2):
    s2_tokenized = [s2_t.text for s2_t in s2 if len(s2_t.text) > 0]
    min_dist = float("inf")
    most_similar = None
    for i, sub in enumerate(s1.sents):
        if len(sub) == 0:
            continue
        sub_tokenized = [sub_t.text for sub_t in sub if len(sub_t.text) > 0]
        dist = Diff(sub_tokenized, s2_tokenized)
        if dist < min_dist:
            min_dist=dist
            most_similar = sub
    return most_similar

def getCoreferences(t1):
    coreferences = []
    for token_idx, token in enumerate(t1):
        if (token.pos_ == 'PRON' and token._.in_coref) or (token.pos_ == 'DET' and token._.in_coref and token.text != 'the' and token.text != 'my'):
            for cluster in token._.coref_clusters:
                # print(token.text + " => " + cluster.main.text)
                coreferences.append(token_idx)
    return coreferences

def test_relationship(t1, t2, corefs):
    if len(corefs) == 0:
        return False

    t1_struct = [t_.pos_ for t_ in t1]
    t2_struct = [t_.pos_ for t_ in t2]
    if len(list(set(t1_struct) - set(t2_struct))) == 0:
        return True

    for ref_idx in corefs:
        t1_coref_pos = t1_struct[ref_idx]
        t2_corresponding = t2_struct[ref_idx]
        if t1_coref_pos == t2_corresponding:
            continue

        t1_after_coref_pos = t1_struct[ref_idx + 1]
        t2_pointer = ref_idx
        while t2_struct[t2_pointer] != t1_after_coref_pos:
            t2_pointer += 1

        t1_struct_compare = t1_struct[ref_idx-2:ref_idx] + t1_struct[ref_idx + 1:]
        t2_struct_compare = t2_struct[ref_idx-2:ref_idx] + t2_struct[t2_pointer:]

        if len(list(set(t1_struct_compare) - set(t2_struct_compare))) == 0:
            return True
    return False



if __name__ == '__main__':
    filename = '/home/jusun/adila001/robust_nlp/cartography/model_output_RTE/overlap_words_analysis/ALL/OOD_CONF_ep_19_SORTED.csv'
    df = pd.read_csv(filename)
    coref_relationships = []
    for i, row in tqdm(df.iterrows()):
        try:
            s1 = row['sentence1'].lower()
            s1 = s1.replace(';', '.')
            s2 = row['sentence2'].lower()
            t1 = nlp(s1)
            t2 = nlp(s2)
            t1 = get_similar_subsentence(t1, t2)

            corefs = getCoreferences(t1)
            coref_relationships.append(test_relationship(t1, t2, corefs))
        except Exception as e:
            print(e)
            coref_relationships.append(None)
            continue
    # print(coref_relationships)
    df['coref_relationship'] = coref_relationships
    df.to_csv(f'{filename[:-4]}_COREF.csv')
    print('NUMBER OF FOUND PATTERN', df.loc[df['coref_relationship']==True].shape[0])
    df_true = df.loc[df['coref_relationship']==True]
    print(df_true)
    df_true.to_csv(f'{filename[:-4]}_COREF_TRUE.csv')
