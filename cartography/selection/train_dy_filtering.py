"""
Filtering and dataset mapping methods based on training dynamics.
By default, this module reads training dynamics from a given trained model and
computes the metrics---confidence, variability, correctness,
as well as baseline metrics of forgetfulness and threshold closeness
for each instance in the training data.
If specified, data maps can be plotted with respect to confidence and variability.
Moreover, datasets can be filtered with respect any of the other metrics.
"""
import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import tqdm
import imageio

from collections import defaultdict
from typing import List

from cartography.data_utils import read_data, read_jsonl, copy_dev_test
from cartography.selection.selection_utils import read_dynamics

# TODO(SS): Named tuple for tasks and filtering methods.

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_forgetfulness(correctness_trend: List[float]) -> int:
  """
  Given a epoch-wise trend of train predictions, compute frequency with which
  an example is forgotten, i.e. predicted incorrectly _after_ being predicted correctly.
  Based on: https://arxiv.org/abs/1812.05159
  """
  if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
      return 1000
  learnt = False  # Predicted correctly in the current epoch.
  times_forgotten = 0
  for is_correct in correctness_trend:
    if (not learnt and not is_correct) or (learnt and is_correct):
      # nothing changed.
      continue
    elif learnt and not is_correct:
      # Forgot after learning at some point!
      learnt = False
      times_forgotten += 1
    elif not learnt and is_correct:
      # Learnt!
      learnt = True
  return times_forgotten


def compute_correctness(trend: List[float]) -> float:
  """
  Aggregate #times an example is predicted correctly during all training epochs.
  """
  return sum(trend)


def compute_train_dy_metrics_per_epoch(training_dynamics, heuristics, original_id, mode="train"):
  """
  Given the training dynamics (logits for each training instance across epochs), compute metrics
  based on it, for data map coorodinates.
  Computed metrics are: confidence, variability, correctness, forgetfulness, threshold_closeness---
  the last two being baselines from prior work
  (Example Forgetting: https://arxiv.org/abs/1812.05159 and
   Active Bias: https://arxiv.org/abs/1704.07433 respectively).
  Returns:
  - DataFrame with these metrics.
  - DataFrame with more typical training evaluation metrics, such as accuracy / loss.
  """
  confidence_ = {}
  variability_ = {}
  threshold_closeness_ = {}
  correctness_ = {}
  forgetfulness_ = {}
  lexical = {}
  constituent = {}
  subsequence= {}
  original_ids = {}
  ood={}
  predicted_labels = {}
  golds_labels = {}

  # Functions to be applied to the data.
  variability_func = lambda conf: np.std(conf)
  # if include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
  #   variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
  threshold_closeness_func = lambda conf: conf * (1 - conf)

  loss = torch.nn.CrossEntropyLoss()
  num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])
  # if burn_out < num_tot_epochs:
  #   logger.info(f"Computing training dynamics. Burning out at {burn_out} of {num_tot_epochs}. ")
  # else:
  logger.info(f"Computing training dynamics across {num_tot_epochs} epochs")
  logger.info("Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness")

  logits = {i: [] for i in range(num_tot_epochs)}
  targets = {i: [] for i in range(num_tot_epochs)}
  training_accuracy = defaultdict(float)

  for guid in tqdm.tqdm(training_dynamics):

    correctness_trend = []
    true_probs_trend = []
    correctness_ep = []
    confidence_ep = []
    variability_ep = []
    prediction_ep = []
    record = training_dynamics[guid]

    for i, epoch_logits in enumerate(record["logits"]):
      if i >= len(logits.keys()):
          break
      probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
      true_class_prob = float(probs[record["gold"]])
      true_probs_trend.append(true_class_prob)

      prediction = np.argmax(epoch_logits)
      is_correct = (prediction == record["gold"]).item()
      correctness_trend.append(is_correct)

      training_accuracy[i] += is_correct
      logits[i].append(epoch_logits)
      targets[i].append(record["gold"])

      correctness_ep.append(compute_correctness(correctness_trend))
      confidence_ep.append(np.mean(true_probs_trend))
      variability_ep.append(variability_func(true_probs_trend))
      prediction_ep.append(prediction.item())
    correctness_[guid] = correctness_ep
    confidence_[guid] = confidence_ep
    variability_[guid] = variability_ep
    # if burn_out < num_tot_epochs:
    #   correctness_trend = correctness_trend[:burn_out]
    #   true_probs_trend = true_probs_trend[:burn_out]

    # correctness_[guid] = compute_correctness(correctness_trend)
    # confidence_[guid] = np.mean(true_probs_trend)
    # variability_[guid] = variability_func(true_probs_trend)

    # forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
    # threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])
    lexical[guid] = heuristics[guid]["lexical"]
    constituent[guid] = heuristics[guid]["constituent"]
    subsequence[guid] = heuristics[guid]["subsequence"]
    ood[guid] = heuristics[guid]["ood"]
    original_ids[guid] = original_id[guid]
    predicted_labels[guid] = prediction_ep

  # Should not affect ranking, so ignoring.
  epsilon_var = np.mean(list(variability_.values()))

  column_names = ['guid',
                  'index',
                  # 'threshold_closeness',
                  'confidence',
                  'variability',
                  'correctness',
                  # 'forgetfulness',
                  'pred_label',
                  'lexical', 'constituent', 'subsequence', 'original_id']
  if mode != "train":
      column_names.insert(-1, "ood")
      df = pd.DataFrame([[guid,
                          i,
                          # threshold_closeness_[guid],
                          confidence_[guid],
                          variability_[guid],
                          correctness_[guid],
                          predicted_labels[guid],
                          # forgetfulness_[guid],
                          lexical[guid],
                          constituent[guid],
                          subsequence[guid],
                          ood[guid],
                          original_ids[guid]
                          ] for i, guid in enumerate(correctness_)], columns=column_names)
      df_train = pd.DataFrame([[i,
                                loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(
                                    training_dynamics),
                                training_accuracy[i] / len(training_dynamics)
                                ] for i in range(num_tot_epochs)],
                              columns=['epoch', 'loss', 'train_acc'])
  else:
      df = pd.DataFrame([[guid,
                          i,
                          # threshold_closeness_[guid],
                          confidence_[guid],
                          variability_[guid],
                          correctness_[guid],
                          predicted_labels[guid],
                          # forgetfulness_[guid],
                          lexical[guid],
                          constituent[guid],
                          subsequence[guid],
                          original_ids[guid]
                          ] for i, guid in enumerate(correctness_)], columns=column_names)
      df_train = pd.DataFrame([[i,loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(training_dynamics),training_accuracy[i] / len(training_dynamics)] for i in range(num_tot_epochs)], columns=['epoch', 'loss', 'train_acc'])
  df.to_csv(f"ALL_SAMPLES_{mode}.csv")
  return df, df_train


def consider_ascending_order(filtering_metric: str) -> bool:
  """
  Determine if the metric values' sorting order to get the most `valuable` examples for training.
  """
  if filtering_metric == "variability":
    return False
  elif filtering_metric == "confidence":
    return True
  elif filtering_metric == "threshold_closeness":
    return False
  elif filtering_metric == "forgetfulness":
    return False
  elif filtering_metric == "correctness":
    return True
  else:
    raise NotImplementedError(f"Filtering based on {filtering_metric} not implemented!")


def write_filtered_data(args, train_dy_metrics):
  """
  Filter data based on the given metric, and write it in TSV format to train GLUE-style classifier.
  """
  # First save the args for filtering, to keep track of which model was used for filtering.
  argparse_dict = vars(args)
  with open(os.path.join(args.filtering_output_dir, f"filtering_configs.json"), "w") as outfile:
    outfile.write(json.dumps(argparse_dict, indent=4, sort_keys=True) + "\n")

  # Determine whether to sort data in ascending order or not, based on the metric.
  is_ascending = consider_ascending_order(args.metric)
  if args.worst:
    is_ascending = not is_ascending

  # Sort by selection.
  sorted_scores = train_dy_metrics.sort_values(by=[args.metric],
                                               ascending=is_ascending)

  original_train_file = os.path.join(os.path.join(args.data_dir, args.task_name), f"train.tsv")
  train_numeric, header = read_data(original_train_file, task_name=args.task_name, guid_as_int=True)

  for fraction in [0.01, 0.05, 0.10, 0.1667, 0.25, 0.3319, 0.50, 0.75]:
    outdir = os.path.join(args.filtering_output_dir,
                          f"cartography_{args.metric}_{fraction:.2f}/{args.task_name}")
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    # Dev and test need not be subsampled.
    copy_dev_test(args.task_name,
                  from_dir=os.path.join(args.data_dir, args.task_name),
                  to_dir=outdir)

    num_samples = int(fraction * len(train_numeric))
    with open(os.path.join(outdir, f"train.tsv"), "w") as outfile:
      outfile.write(header + "\n")
      selected = sorted_scores.head(n=num_samples+1)
      if args.both_ends:
        hardest = sorted_scores.head(n=int(num_samples * 0.7))
        easiest = sorted_scores.tail(n=num_samples - hardest.shape[0])
        selected = pd.concat([hardest, easiest])
        fm = args.metric
        logger.info(f"Selecting both ends: {fm} = "
                    f"({hardest.head(1)[fm].values[0]:3f}: {hardest.tail(1)[fm].values[0]:3f}) "
                    f"& ({easiest.head(1)[fm].values[0]:3f}: {easiest.tail(1)[fm].values[0]:3f})")

      selection_iterator = tqdm.tqdm(range(len(selected)))
      for idx in selection_iterator:
        selection_iterator.set_description(
          f"{args.metric} = {selected.iloc[idx][args.metric]:.4f}")

        selected_id = selected.iloc[idx]["guid"]
        if args.task_name in ["SNLI", "MNLI"]:
          selected_id = int(selected_id)
        elif args.task_name == "WINOGRANDE":
          selected_id = str(int(selected_id))
        record = train_numeric[selected_id]
        outfile.write(record + "\n")

    logger.info(f"Wrote {num_samples} samples to {outdir}.")

def mix_heuristics_label_eval(df):
    df_lex_supp = df.loc[(df["lexical"] == 1)& (df["constituent"] == 0) &(df["subsequence"] == 0)]
    df_lex_supp['mix_heurstic_label'] = f"lexical support (ood: " \
                                        f"{df_lex_supp.loc[df_lex_supp['ood']==1].shape[0]} id: {df_lex_supp.loc[df_lex_supp['ood']==0].shape[0]})"
    df_lex_cont = df.loc[(df["lexical"] == -1) & (df["constituent"] == 0) & (df["subsequence"] == 0)]
    df_lex_cont['mix_heurstic_label'] = f"lexical contradict (ood: " \
                                        f"{df_lex_cont.loc[df_lex_cont['ood']==1].shape[0]} id: {df_lex_cont.loc[df_lex_cont['ood']==0].shape[0]})"
    df_const_supp = df.loc[(df["lexical"] == 0) & (df["constituent"] == 1) & (df["subsequence"] == 0)]
    df_const_supp['mix_heurstic_label'] = f"constituent support (ood: " \
                                        f"{df_const_supp.loc[df_const_supp['ood']==1].shape[0]} id: {df_const_supp.loc[df_const_supp['ood']==0].shape[0]})"
    df_const_cont = df.loc[(df["lexical"] == 0) & (df["constituent"] == -1) & (df["subsequence"] == 0)]
    df_const_cont['mix_heurstic_label'] = f"constituent contradict (ood: " \
                                        f"{df_const_cont.loc[df_const_cont['ood']==1].shape[0]} id: {df_const_cont.loc[df_const_cont['ood']==0].shape[0]})"
    df_sub_supp = df.loc[(df["lexical"] == 0) & (df["constituent"] == 0) & (df["subsequence"] == 1)]
    df_sub_supp['mix_heurstic_label'] = f"subsequence support (ood: " \
                                        f"{df_sub_supp.loc[df_sub_supp['ood']==1].shape[0]} id: {df_sub_supp.loc[df_sub_supp['ood']==0].shape[0]})"
    df_sub_cont = df.loc[(df["lexical"] == 0) & (df["constituent"] == 0) & (df["subsequence"] == -1)]
    df_sub_cont['mix_heurstic_label'] = f"subsequence contradict (ood: " \
                                        f"{df_sub_cont.loc[df_sub_cont['ood']==1].shape[0]} id: {df_sub_cont.loc[df_sub_cont['ood']==0].shape[0]})"
    df_mix = pd.concat([df_lex_supp, df_lex_cont, df_const_supp, df_const_cont,
                    df_sub_supp, df_sub_cont])
    return df_mix

def mix_heuristics_label_train(df):
    df_lex_supp = df.loc[(df["lexical"] == 1)& (df["constituent"] == 0) &(df["subsequence"] == 0)]
    df_lex_supp['mix_heurstic_label'] = f"lexical support ({df_lex_supp.shape[0]}"
    df_lex_cont = df.loc[(df["lexical"] == -1) & (df["constituent"] == 0) & (df["subsequence"] == 0)]
    df_lex_cont['mix_heurstic_label'] = f"lexical contradict ({df_lex_cont.shape[0]}"
    df_const_supp = df.loc[(df["lexical"] == 0) & (df["constituent"] == 1) & (df["subsequence"] == 0)]
    df_const_supp['mix_heurstic_label'] = f"constituent support ({df_const_supp.shape[0]}"
    df_const_cont = df.loc[(df["lexical"] == 0) & (df["constituent"] == -1) & (df["subsequence"] == 0)]
    df_const_cont['mix_heurstic_label'] = f"constituent contradict ({df_const_cont.shape[0]}"
    df_sub_supp = df.loc[(df["lexical"] == 0) & (df["constituent"] == 0) & (df["subsequence"] == 1)]
    df_sub_supp['mix_heurstic_label'] = f"subsequence support ({df_sub_supp.shape[0]}"
    df_sub_cont = df.loc[(df["lexical"] == 0) & (df["constituent"] == 0) & (df["subsequence"] == -1)]
    df_sub_cont['mix_heurstic_label'] = f"subsequence contradict ({df_sub_cont.shape[0]}"
    df_mix = pd.concat([df_lex_supp, df_lex_cont, df_const_supp, df_const_cont,
                    df_sub_supp, df_sub_cont])
    return df_mix

def get_ambiguous_heuristics_samples(df, model_dir, df_orig=pd.read_csv("/home/jusun/adila001/MNLI/train_heuristic.tsv", sep='\t|\n'),
                                     heu='lexical'):
    df = df.loc[(df[heu] != 0)]


    df = df.loc[df["variability"] >= 0.3]
    df_heuristics_ORIG = df_orig.loc[df['original_id'].tolist()]
    df_heuristics_ORIG= df_heuristics_ORIG.drop(['index', 'promptID', 'pairID'], axis=1)
    df_heuristics_ORIG['confidence'] = df['confidence'].tolist()
    df_heuristics_ORIG['variability'] = df['variability'].tolist()
    csv_dir = os.path.join(model_dir, 'unique_samples_csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    df_heuristics_ORIG.to_csv(os.path.join(csv_dir, 'ambiguous_samples.csv'))


def get_sorted_samples(df, model_dir, df_orig=pd.read_csv('/home/jusun/adila001/MNLI/train_heuristic.tsv', sep='\t|\n'),
                                 n_sample=30,
                                 decoded_label=["contradiction", "entailment", "neutral"],
                                 columns_order = ['index', 'genre', 'sentence1', 'sentence2', 'variability',
                                                  'confidence', 'var_ep', 'conf_ep', 'gold_label', 'pred_label'], mode="train"):
    csv_dir = os.path.join(model_dir, 'ANALYSIS_CLEAN', 'SORTED')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    # heuristics = top_heuristic_obj.keys()
    ep_number = len(df['variability'].tolist()[0])
    for ep in range(ep_number):
        # for heu in heuristics:
        #     df_heuristic = df.loc[df[heu] != 0]
        df_copy = df.copy()
        df_copy['var_ep'] = np.asarray(df_copy['variability'].tolist())[:, ep]
        df_copy['conf_ep'] = np.asarray(df_copy['confidence'].tolist())[:, ep]
        df_copy['pred_label'] = np.asarray(df_copy['pred_label'].tolist())[:, ep]
        df_copy['pred_label'] = [decoded_label[pred] for pred in df_copy['pred_label']]
        # random_sample = df_heuristic.sample(n= top_heuristic_obj[heu])
        # top_n_var = df_copy.nlargest(n_sample, 'var_ep')
        # top_n_conf = df_copy.nsmallest(df_copy.shape[0], 'conf_ep')
        top_n_conf = df_copy.sort_values(by=['conf_ep'])

        # top_n_var_ORIG = df_orig.loc[top_n_var['original_id'].tolist()]
        # # top_n_var_ORIG = top_n_var_ORIG.drop(cols_to_drop, axis=1)
        # top_n_var_ORIG['variability'] = top_n_var['variability'].tolist()
        # top_n_var_ORIG['confidence'] = top_n_var['confidence'].tolist()
        # top_n_var_ORIG['var_ep'] = top_n_var['var_ep'].tolist()
        # top_n_var_ORIG['conf_ep'] = top_n_var['conf_ep'].tolist()
        # top_n_var_ORIG['pred_label'] = top_n_var['pred_label'].tolist()

        top_n_conf_ORIG = df_orig.loc[top_n_conf['original_id'].tolist()]
        # top_n_conf_ORIG = top_n_conf_ORIG.drop(cols_to_drop, axis=1)
        top_n_conf_ORIG['variability'] = top_n_conf['variability'].tolist()
        top_n_conf_ORIG['confidence'] = top_n_conf['confidence'].tolist()
        top_n_conf_ORIG['var_ep'] = top_n_conf['var_ep'].tolist()
        top_n_conf_ORIG['conf_ep'] = top_n_conf['conf_ep'].tolist()
        top_n_conf_ORIG['pred_label'] = top_n_conf['pred_label'].tolist()

        # top_n_var_ORIG = top_n_var_ORIG[columns_order]
        top_n_conf_ORIG = top_n_conf_ORIG[columns_order]

        prefix = mode.upper()
        # top_n_var_ORIG.to_csv(os.path.join(csv_dir, "{}_SORTED_VAR_ep_{}.csv".format(prefix, ep)))
        top_n_conf_ORIG.to_csv(os.path.join(csv_dir, "{}_CONF_ep_{}_SORTED.csv".format(prefix, ep)))

    # return top_n_var_ORIG, top_n_conf_ORIG

def get_top_n_heuristics_samples(df, model_dir, df_orig=pd.read_csv('/home/jusun/adila001/MNLI/train_heuristic.tsv', sep='\t|\n'),
                                 top_heuristic_obj = {'lexical': 20, 'constituent': 20, 'subsequence': 20},
                                 decoded_label=["contradiction", "entailment", "neutral"],
                                 columns_order = ['genre', 'sentence1', 'sentence2', 'variability',
                                                  'confidence', 'var_ep', 'conf_ep', 'gold_label', 'pred_label']):
    csv_dir = os.path.join(model_dir, 'heuristics_only_csv_EVAL')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    heuristics = top_heuristic_obj.keys()
    ep_number = len(df['variability'].tolist()[0])
    for ep in range(ep_number):
        for heu in heuristics:
            df_heuristic = df.loc[df[heu] != 0]
            df_heuristic['var_ep'] = np.asarray(df_heuristic['variability'].tolist())[:,ep]
            df_heuristic['conf_ep'] = np.asarray(df_heuristic['confidence'].tolist())[:, ep]
            df_heuristic['pred_label'] = np.asarray(df_heuristic['pred_label'].tolist())[:, ep]
            df_heuristic['pred_label'] = [decoded_label[pred] for pred in df_heuristic['pred_label']]
            # random_sample = df_heuristic.sample(n= top_heuristic_obj[heu])
            top_n_var = df_heuristic.nlargest(top_heuristic_obj[heu],'var_ep')
            top_n_conf = df_heuristic.nlargest(top_heuristic_obj[heu],'conf_ep')

            # random_sample_ORIG = df_orig.loc[random_sample['original_id'].tolist()]
            # random_sample_ORIG = random_sample_ORIG.drop(['index', 'promptID', 'pairID'], axis=1)
            # random_sample_ORIG['variability'] = random_sample['variability'].tolist()
            # random_sample_ORIG['confidence'] = random_sample['confidence'].tolist()
            # random_sample_ORIG['var_ep'] = random_sample['var_ep'].tolist()
            # random_sample_ORIG['conf_ep'] = random_sample['conf_ep'].tolist()

            top_n_var_ORIG = df_orig.loc[top_n_var['original_id'].tolist()]
            # top_n_var_ORIG = top_n_var_ORIG.drop(cols_to_drop, axis=1)
            top_n_var_ORIG['variability'] = top_n_var['variability'].tolist()
            top_n_var_ORIG['confidence'] = top_n_var['confidence'].tolist()
            top_n_var_ORIG['var_ep'] = top_n_var['var_ep'].tolist()
            top_n_var_ORIG['conf_ep'] = top_n_var['conf_ep'].tolist()
            top_n_var_ORIG['pred_label'] = top_n_var['pred_label'].tolist()

            top_n_conf_ORIG = df_orig.loc[top_n_conf['original_id'].tolist()]
            # top_n_conf_ORIG = top_n_conf_ORIG.drop(cols_to_drop, axis=1)
            top_n_conf_ORIG['variability'] = top_n_conf['variability'].tolist()
            top_n_conf_ORIG['confidence'] = top_n_conf['confidence'].tolist()
            top_n_conf_ORIG['var_ep'] = top_n_conf['var_ep'].tolist()
            top_n_conf_ORIG['conf_ep'] = top_n_conf['conf_ep'].tolist()
            top_n_conf_ORIG['pred_label'] = top_n_conf['pred_label'].tolist()

            top_n_var_ORIG = top_n_var_ORIG[columns_order]
            top_n_conf_ORIG = top_n_conf_ORIG[columns_order]
            # print(random_sample_ORIG)
            # print(f"{heu}_EP_{ep}")
            # print(top_n_var_ORIG)
            # print(top_n_conf_ORIG)

            # top_n_var_ORIG.to_csv(os.path.join(csv_dir, "{}_TOP_VAR_ep_{}.csv".format(heu, ep)))
            top_n_conf_ORIG.to_csv(os.path.join(csv_dir, "{}_TOP_CONF_ep_{}_LARGEST.csv".format(heu, ep)))

        # return top_n_var_ORIG, top_n_conf_ORIG
        # random_sample_ORIG.to_csv(os.path.join(csv_dir,"{}_RANDOM.csv".format(heu)),index=False)
        # top_n_var_ORIG.to_csv(os.path.join(csv_dir, "{}_TOP_VAR.csv".format(heu)), index=False)
        # top_n_conf_ORIG.to_csv(os.path.join(csv_dir, "{}_TOP_CONF.csv".format(heu)), index=False)

def find_max_var(var_arr):
    return np.amax(var_arr)

def plot_train_epochs(args, training_dynamics, heuristics, original_id, gif =True):
    total_epochs = len(list(training_dynamics.values())[0]["logits"])
    df, _ = compute_train_dy_metrics_per_epoch(training_dynamics, heuristics, original_id)
    train_dy_filename = os.path.join(args.model_dir, f"td_metrics.jsonl")
    df.to_json(train_dy_filename,
                             orient='records',
                             lines=True)
    logger.info(f"Metrics based on Training Dynamics written to {train_dy_filename}")
    df_heuristics = df.loc[(df["lexical"] != 0) | (df["constituent"] != 0) | (df["subsequence"] != 0)]
    df_others = df.loc[(df["lexical"] == 0) & (df["constituent"] == 0) & (df["subsequence"] == 0)]
    max_instances_heuristic = {
        'lexical': df_heuristics.loc[df_heuristics['lexical'] != 0].shape[0],
        'subsequence': df_heuristics.loc[df_heuristics['subsequence'] != 0].shape[0],
        'constituent': df_heuristics.loc[df_heuristics['constituent'] != 0].shape[0]
    }
    heuristics = ['lexical', 'constituent', 'subsequence']

    max_var = find_max_var(df_heuristics['variability'].tolist())

    for heuristic in heuristics:
        figs = []
        max_instances_to_plot = max_instances_heuristic[heuristic]
        df_current_heuristic = df_heuristics.loc[df_heuristics[heuristic] != 0]
        # df_others_sampled = df_others.sample(n=max_instances_to_plot-df_current_heuristic.shape[0])
        df_others_sampled = df_others.sample(n=df_current_heuristic.shape[0]*2)
        df_current_heuristic = df_current_heuristic.append(df_others_sampled, ignore_index=True)

        # ### DEV ###
        # for ep in range(total_epochs):
        # ### DEV ###
        for ep in range(2,total_epochs):
            df_current_heuristic_epoch = df_current_heuristic.copy()
            # print(df_current_heuristic_epoch['confidence'])
            confidence_epoch = np.asarray(df_current_heuristic_epoch['confidence'].tolist())[:,ep].flatten()
            var_epoch = np.asarray(df_current_heuristic_epoch['variability'].tolist())[:,ep].flatten()
            correctness_epoch = np.asarray(df_current_heuristic_epoch['correctness'].tolist())[:,ep].flatten()
            df_current_heuristic_epoch.drop(['confidence', 'variability', 'correctness'], axis=1)
            df_current_heuristic_epoch['confidence'] = confidence_epoch
            df_current_heuristic_epoch['variability'] = var_epoch
            df_current_heuristic_epoch['correctness'] = correctness_epoch
            fig = plot_heuristics_mix(df_current_heuristic_epoch, os.path.join(args.plots_dir, 'train_plots'), hue_metric=heuristic,
                                title='{}_epoch_{}'.format(heuristic, ep), max_var=max_var)
            figs.append(convert_fig_to_arr(fig))
        if gif:
            kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
            gif_path = os.path.join(args.plots_dir, "train_plots", f'TRAIN_{ep}_epochs.gif')
            # gif_path = f'{args.plots_dir}/{heuristic}_{ep}_epochs.gif'
            imageio.mimsave(gif_path, figs, fps=1)
            logger.info(f"Aminated gif saved to {gif_path}")

    df_heuristics_mix = mix_heuristics_label_train(df_heuristics)
    figs = []
    # ### DEV ###
    # for ep in range(total_epochs):
    # ### DEV ###
    df_others_sampled = df_others.sample(n=df_heuristics_mix.shape[0] * 2)
    for ep in range(2,total_epochs):
        df_heuristic_mix_epoch = df_heuristics_mix.copy()
        confidence_epoch = np.asarray(df_heuristic_mix_epoch['confidence'].tolist())[:, ep].flatten()
        var_epoch = np.asarray(df_heuristic_mix_epoch['variability'].tolist())[:, ep].flatten()
        correctness_epoch = np.asarray(df_heuristic_mix_epoch['correctness'].tolist())[:, ep].flatten()
        df_heuristic_mix_epoch.drop(['confidence', 'variability', 'correctness'], axis=1)
        df_heuristic_mix_epoch['confidence'] = confidence_epoch
        df_heuristic_mix_epoch['variability'] = var_epoch
        df_heuristic_mix_epoch['correctness'] = correctness_epoch
        # df_heuristic_mix_epoch = pd.concat([df_others_sampled, df_heuristic_mix_epoch])


        fig = plot_heuristics_only(df_heuristic_mix_epoch,os.path.join(args.plots_dir, "train_plots"),title=f'HEURISTICS_ONLY_{ep}', max_var=max_var)
        figs.append(convert_fig_to_arr(fig))
    if gif:
        kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
        gif_path = os.path.join(args.plots_dir, "train_plots", f'TRAIN_{ep}_epochs_ALL.gif')
        # gif_path = f'{args.plots_dir}/HEURISTICS_ONLY_{ep}_epochs.gif'
        imageio.mimsave(gif_path, figs, fps=1)
        logger.info(f"Aminated gif saved to {gif_path}")

def plot_eval_epochs(args, id_obj, ood_obj, gif =True):
    id_dynamics, id_heuristics, id_original_idx, id_pred = id_obj[0], id_obj[1], id_obj[2], id_obj[3]
    ood_dynamics, ood_heuristics, ood_original_idx, ood_pred = ood_obj[0], ood_obj[1], ood_obj[2], ood_obj[3]
    total_epochs = len(list(id_dynamics.values())[0]["logits"])
    df_id, _ = compute_train_dy_metrics_per_epoch(id_dynamics, id_heuristics, id_original_idx, mode="eval")
    df_ood, _ = compute_train_dy_metrics_per_epoch(ood_dynamics, ood_heuristics, ood_original_idx, mode="eval")
    df_ood['ood'] = 1
    df_id['ood'] = 0
    id_dy_filename = os.path.join(args.model_dir, f"iid_metrics.jsonl")
    df_id.to_json(id_dy_filename,
               orient='records',
               lines=True)
    ood_dy_filename = os.path.join(args.model_dir, f"ood_metrics.jsonl")
    df_ood.to_json(ood_dy_filename,
                  orient='records',
                  lines=True)
    logger.info(f"Metrics based on Eval Dynamics written to {id_dy_filename} and {ood_dy_filename}")

    df = pd.concat([df_id, df_ood])

    max_var = find_max_var(df['variability'].tolist())
    df_heuristics = df.loc[(df["lexical"] != 0) | (df["constituent"] != 0) | (df["subsequence"] != 0)]
    df_ood = df.loc[(df["ood"] != 0)]

    df_concern = pd.concat([df_heuristics, df_ood])
    df_concern = mix_heuristics_label_eval(df_concern)

    df_others = df.loc[(df["lexical"] == 0) & (df["constituent"] == 0) & (df["subsequence"] == 0) & (df["ood"] == 0)]
    print(df_others.shape)
    print(df_ood.shape)
    df_others_sample = df_others.sample(n= int(np.ceil(df_ood.shape[0])) if df_ood.shape[0] < df_others.shape[0] else df_others.shape[0]  )
    df = pd.concat([df_concern, df_others_sample])
    df = df.fillna("no heuristic")

    print(df_heuristics.shape[0], df_others_sample.shape[0], df_ood.shape[0])

    figs = []
    palette=iter(sns.husl_palette(len(np.unique(df["mix_heurstic_label"].tolist()))+1))
    # ### DEV ###
    # for ep in range(total_epochs):
    # ### DEV ###
    for ep in range(2, total_epochs):
        df_heuristic_mix_epoch = df.copy()
        confidence_epoch = np.asarray(df_heuristic_mix_epoch['confidence'].tolist())[:, ep].flatten()
        var_epoch = np.asarray(df_heuristic_mix_epoch['variability'].tolist())[:, ep].flatten()
        correctness_epoch = np.asarray(df_heuristic_mix_epoch['correctness'].tolist())[:, ep].flatten()
        df_heuristic_mix_epoch.drop(['confidence', 'variability', 'correctness'], axis=1)
        df_heuristic_mix_epoch['confidence'] = confidence_epoch
        df_heuristic_mix_epoch['variability'] = var_epoch
        df_heuristic_mix_epoch['correctness'] = correctness_epoch

        fig = plot_heuristics_only(df_heuristic_mix_epoch, os.path.join(args.plots_dir, "eval_plots"), title=f'EVAL_{ep}',
                                   max_var=max_var, style="ood")
        figs.append(convert_fig_to_arr(fig))
    if gif:
        kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
        gif_path = os.path.join(args.plots_dir, "eval_plots", f'EVAL_{ep}_epochs.gif')
        imageio.mimsave(gif_path, figs, fps=1)
        logger.info(f"Aminated gif saved to {gif_path}")


def convert_fig_to_arr(fig):
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

def compute_train_dy_metrics(training_dynamics, heuristics, original_id, burn_out):
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}
    lexical = {}
    constituent = {}
    subsequence = {}
    original_ids = {}

    # Functions to be applied to the data.
    variability_func = lambda conf: np.std(conf)
    threshold_closeness_func = lambda conf: conf * (1 - conf)
    loss = torch.nn.CrossEntropyLoss()
    num_tot_epochs = len(list(training_dynamics.values())[0]["logits"])

    logger.info(f"Computing training dynamics across {num_tot_epochs} epochs")
    logger.info("Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness")

    logits = {i: [] for i in range(num_tot_epochs)}
    targets = {i: [] for i in range(num_tot_epochs)}
    training_accuracy = defaultdict(float)

    for guid in tqdm.tqdm(training_dynamics):
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics[guid]
        for i, epoch_logits in enumerate(record["logits"]):
            if i >= len(logits.keys()):
                break
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[record["gold"]])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record["gold"]).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record["gold"])

        if burn_out < num_tot_epochs:
            correctness_trend = correctness_trend[:burn_out]
            true_probs_trend = true_probs_trend[:burn_out]

        correctness_[guid] = compute_correctness(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        variability_[guid] = variability_func(true_probs_trend)

        forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])
        lexical[guid] = heuristics[guid]["lexical"]
        constituent[guid] = heuristics[guid]["constituent"]
        subsequence[guid] = heuristics[guid]["subsequence"]
        original_ids[guid] = original_id[guid]

    # Should not affect ranking, so ignoring.
    epsilon_var = np.mean(list(variability_.values()))

    column_names = ['guid',
                    'index',
                    'threshold_closeness',
                    'confidence',
                    'variability',
                    'correctness',
                    'forgetfulness', 'lexical', 'constituent', 'subsequence', 'original_id']
    df = pd.DataFrame([[guid,
                        i,
                        threshold_closeness_[guid],
                        confidence_[guid],
                        variability_[guid],
                        correctness_[guid],
                        forgetfulness_[guid],
                        lexical[guid],
                        constituent[guid],
                        subsequence[guid],
                        original_ids[guid]
                        ] for i, guid in enumerate(correctness_)], columns=column_names)
    df_train = pd.DataFrame([[i,
                              loss(torch.Tensor(logits[i]), torch.LongTensor(targets[i])).item() / len(
                                  training_dynamics),
                              training_accuracy[i] / len(training_dynamics)
                              ] for i in range(num_tot_epochs)],
                            columns=['epoch', 'loss', 'train_acc'])
    return df, df_train

def plot_heuristics_only(
                  df: pd.DataFrame,
                  plot_dir: os.path,
                  title: str = '', save=True, max_var=0.5, style=None, palette = None):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    main_metric = 'variability'
    other_metric = 'confidence'

    hue = "mix_heurstic_label"

    num_hues = len(df[hue].unique().tolist())

    fig, ax0 = plt.subplots(1, 1, figsize=(12, 10))
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")
    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=df,
                           hue=hue,
                           # palette=pal,
                           # style=hue,
                           s=30,
                           style=style,
                           marker = 'o' if style is not None else None,
                           # palette="tab10"
                           # palette=['green', 'orange', 'brown', 'dodgerblue', 'red']
                           palette = palette if palette else "tab10"
                           )

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda text, xyc, bbc: ax0.annotate(text,
                                                        xy=xyc,
                                                        xycoords="axes fraction",
                                                        fontsize=15,
                                                        color='black',
                                                        va="center",
                                                        ha="center",
                                                        rotation=350,
                                                        bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')

    plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right', fontsize ='small')
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')
    plot.set_title(title)
    ax0.set_xlim(0, max_var)
    ax0.set_ylim(0, 1)


    fig.tight_layout()
    filename = f'{plot_dir}/{title}.png'
    if save:
        fig.savefig(filename, dpi=300)
        logger.info(f"Plot saved to {filename}")
    return fig

def plot_heuristics_mix(
                  df: pd.DataFrame,
                  plot_dir: os.path,
                  hue_metric: str = 'lexical',
                  title: str = '',
                  save=True,
                  max_var = 0.5):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    # Normalize correctness to a value between 0 and 1.
    dataframe = df.assign(corr_frac=lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric

    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=df,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda text, xyc, bbc: ax0.annotate(text,
                                                        xy=xyc,
                                                        xycoords="axes fraction",
                                                        fontsize=15,
                                                        color='black',
                                                        va="center",
                                                        ha="center",
                                                        rotation=350,
                                                        bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')

    plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')
    plot.set_title(title)
    ax0.set_xlim(0, max_var)
    ax0.set_ylim(0, 1)

    fig.tight_layout()
    filename = f'{plot_dir}/{title}.png'
    # print('PLOT HIST', filename)
    if save:
        fig.savefig(filename, dpi=300)
        logger.info(f"Plot saved to {filename}")
    return fig


def plot_data_map(dataframe: pd.DataFrame,
                  plot_dir: os.path,
                  hue_metric: str = 'correct.',
                  title: str = '',
                  model: str = 'RoBERTa',
                  show_hist: bool = False,
                  max_instances_to_plot = 55000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')
    logger.info(f"Plotting figure for {title} using the {model} model ...")

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    if show_hist:
        plot.set_title(f"{title}-{model} Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')
        plott1[0].set_ylabel('density')

        plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True) # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

    fig.tight_layout()
    filename = f'{plot_dir}/{title}_{model}.pdf' if show_hist else f'figures/compact_{title}_{model}.png'
    print('PLOT ORIGINAL', filename)
    fig.savefig(filename, dpi=300)
    logger.info(f"Plot saved to {filename}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--filter",
                      action="store_true",
                      help="Whether to filter data subsets based on specified `metric`.")
  parser.add_argument("--plot_train",
                      action="store_true",
                      help="Whether to plot train data maps and save as `png`.")
  parser.add_argument("--plot_eval",
                      action="store_true",
                      help="Whether to plot eval data maps and save as `png`.")
  parser.add_argument("--model_dir",
                      "-o",
                      required=True,
                      type=os.path.abspath,
                      help="Directory where model training dynamics stats reside.")
  parser.add_argument("--data_dir",
                      "-d",
                      default="/Users/swabhas/data/glue/WINOGRANDE/xl/",
                      type=os.path.abspath,
                      help="Directory where data for task resides.")
  parser.add_argument("--plots_dir",
                      default="./cartography/",
                      type=os.path.abspath,
                      help="Directory where plots are to be saved.")
  parser.add_argument("--task_name",
                      "-t",
                      default="WINOGRANDE",
                      choices=("SNLI", "MNLI", "QNLI", "WINOGRANDE", "RTE", "WNLI"),
                      help="Which task are we plotting or filtering for.")
  parser.add_argument('--metric',
                      choices=('threshold_closeness',
                               'confidence',
                               'variability',
                               'correctness',
                               'forgetfulness'),
                      help="Metric to filter data by.",)
  parser.add_argument("--include_ci",
                      action="store_true",
                      help="Compute the confidence interval for variability.")
  parser.add_argument("--filtering_output_dir",
                      "-f",
                      default="./filtered/",
                      type=os.path.abspath,
                      help="Output directory where filtered datasets are to be written.")
  parser.add_argument("--worst",
                      action="store_true",
                      help="Select from the opposite end of the spectrum acc. to metric,"
                           "for baselines")
  parser.add_argument("--both_ends",
                      action="store_true",
                      help="Select from both ends of the spectrum acc. to metric,")
  parser.add_argument("--burn_out",
                      type=int,
                      default=100,
                      help="# Epochs for which to compute train dynamics.")
  parser.add_argument("--model",
                      default="RoBERTa",
                      help="Model for which data map is being plotted")
  parser.add_argument("--plot_gif",
                      action="store_true",
                      help="Whether to plot gif or not")

  args = parser.parse_args()
  # if args.plot_gif:
  #   assert len(os.listdir(args.plots_dir)) > 0
  #   plot_gif(args.plots_dir)
  #   exit()
  # total_epochs = len(list(training_dynamics.values())[0]["logits"])
  # if args.burn_out > total_epochs:
  #   args.burn_out = total_epochs
  #   logger.info(f"Total epochs found: {args.burn_out}")
  # train_dy_metrics, _ = compute_train_dy_metrics(training_dynamics, heuristics, original_id, args.burn_out)
  #
  # burn_out_str = f"_{args.burn_out}" if args.burn_out > total_epochs else ""
  # train_dy_filename = os.path.join(args.model_dir, f"td_metrics{burn_out_str}.jsonl")
  # train_dy_metrics.to_json(train_dy_filename,
  #                          orient='records',
  #                          lines=True)
  # logger.info(f"Metrics based on Training Dynamics written to {train_dy_filename}")

  # if args.filter:
  #   assert args.filtering_output_dir
  #   if not os.path.exists(args.filtering_output_dir):
  #     os.makedirs(args.filtering_output_dir)
  #   assert args.metric
  #   write_filtered_data(args, train_dy_metrics)

  assert args.plots_dir
  args.plots_dir = os.path.join(args.model_dir, "plots")
  if not os.path.exists(args.plots_dir):
      os.makedirs(args.plots_dir)

  if args.plot_train:
    # plot_data_map(train_dy_metrics, args.plots_dir, title=args.task_name, show_hist=True, model=args.model)
    # plot_heuristics_mix(args, train_dy_metrics, args.plots_dir, title=args.task_name)
    # plot_heuristics_only(args, train_dy_metrics, args.plots_dir, title=args.task_name)
    # get_ambiguous_heuristics_samples(train_dy_metrics, args.model_dir)
    # get_top_n_heuristics_samples(train_dy_metrics, args.model_dir)
    training_dynamics, heuristics, original_id, pred_labels = read_dynamics(args.model_dir,
                                                                            strip_last=True if args.task_name in [
                                                                                "QNLI"] else False,
                                                                            burn_out=args.burn_out if args.burn_out < 100 else None)
    df_train, _ = compute_train_dy_metrics_per_epoch(training_dynamics, heuristics, original_id, mode="eval")
    # plot_train_epochs(args, training_dynamics, heuristics, original_id, gif=True)
    get_sorted_samples(df_train, args.model_dir,
                       pd.read_csv('/home/jusun/adila001/RTE/train_heuristic.tsv',
                                   sep='\t|\n'),
                       decoded_label=["entailment", "not_entailment"],
                       columns_order=['index', 'sentence1', 'sentence2', 'variability',
                                      'confidence', 'var_ep', 'conf_ep', 'lexical',
                                      'subsequence',
                                      'gold_label', 'pred_label'])
    # get_top_n_heuristics_samples(df_train, args.model_dir,
    #                                                pd.read_csv('/home/jusun/adila001/RTE/train_heuristic.tsv',
    #                                                            sep='\t|\n'),
    #                                                columns_order=['sentence1', 'sentence2', 'variability',
    #                                                               'confidence', 'var_ep', 'conf_ep', 'lexical',
    #                                                               'subsequence',
    #                                                               'gold_label', 'pred_label'],
    #                                                 decoded_label=["entailment", "not_entailment"],
    #                                                top_heuristic_obj={'lexical': 10})

  if args.plot_eval:
      # get_ambiguous_heuristics_samples(train_dy_metrics, args.model_dir)
      eval_ID_dynamics, heuristics_ID, original_id_ID, pred_labels_ID = read_dynamics(args.model_dir,
                                                                              strip_last=True if args.task_name in [
                                                                                  "QNLI"] else False,
                                                                              burn_out=args.burn_out if args.burn_out < 100 else None, mode="eval_ID")
      eval_OOD_dynamics, heuristics_OOD, original_id_OOD, pred_labels_OOD = read_dynamics(args.model_dir,
                                                                             strip_last=True if args.task_name in [
                                                                                 "QNLI"] else False,
                                                                             burn_out=args.burn_out if args.burn_out < 100 else None, mode="eval_OOD")
      df_id, _ = compute_train_dy_metrics_per_epoch(eval_ID_dynamics, heuristics_ID, original_id_ID, mode="in_dist")
      df_ood, _ = compute_train_dy_metrics_per_epoch(eval_OOD_dynamics, heuristics_OOD, original_id_OOD, mode="ood")
      # get_top_n_heuristics_samples(df_id, args.model_dir,
      #                              pd.read_csv('/home/jusun/adila001/MNLI/dev_matched_heuristic.tsv', sep='\t|\n'),
      #                              # decoded_label=["entailment", "not_entailment"],
      #                              columns_order=['sentence1', 'sentence2', 'variability',
      #                                             'confidence', 'var_ep', 'conf_ep', 'lexical', 'constituent',
      #                                             'subsequence',
      #                                             'gold_label', 'pred_label'],
      #                              top_heuristic_obj={'lexical': 30})
      # # df_ood['ood'] = 1
      # get_top_n_heuristics_samples(df_ood, args.model_dir,
      #                              pd.read_csv('/home/jusun/adila001/WNLI/train_heuristic.tsv', sep='\t|\n'),
      #                              decoded_label=["not_entailment", "entailment"],
      #                              columns_order=['sentence1', 'sentence2', 'variability',
      #                                             'confidence', 'var_ep', 'conf_ep', 'lexical', 'subsequence',
      #                                             'gold_label', 'pred_label'],
      #                              top_heuristic_obj={'lexical':30})
      get_sorted_samples(df_id, args.model_dir,
                                                     pd.read_csv('/home/jusun/adila001/RTE/dev_heuristic.tsv',
                                                                 sep='\t|\n'),
                                                     decoded_label=["entailment", "not_entailment"],
                                                     columns_order=['index', 'sentence1', 'sentence2', 'variability',
                                                                    'confidence', 'var_ep', 'conf_ep', 'lexical',
                                                                    'subsequence',
                                                                    'gold_label', 'pred_label'], mode='in_dist')
      df_ood['ood'] = 1
      get_sorted_samples(df_ood, args.model_dir,
                                                       pd.read_csv('/home/jusun/adila001/WNLI/train_heuristic.tsv',
                                                                   sep='\t|\n'),
                                                       decoded_label=["not_entailment", "entailment"],
                                                       columns_order=['index', 'sentence1', 'sentence2', 'variability',
                                                                      'confidence', 'var_ep', 'conf_ep', 'lexical',
                                                                      'subsequence',
                                                                      'gold_label', 'pred_label'], mode='ood')
      # print(id_conf)
      # print(ood_conf)
      plot_eval_epochs(args, [eval_ID_dynamics, heuristics_ID, original_id_ID, pred_labels_ID],
                       [eval_OOD_dynamics, heuristics_OOD, original_id_OOD, pred_labels_OOD], gif=True)
    
