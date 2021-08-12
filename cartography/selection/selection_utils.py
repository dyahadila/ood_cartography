import json
import logging
import numpy as np
import os
import pandas as pd
import tqdm

from typing import List

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def log_training_dynamics(output_dir: os.path,
                          epoch: int,
                          train_ids: List[int],
                          train_logits: List[List[float]],
                          train_golds: List[int],
                          heuristics: List[List[int]],
                          original_ids: List[int],
                          predictions: List[int]):
  """
  Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
  """
  heuristics=heuristics.tolist()
  td_df = pd.DataFrame({"guid": train_ids,
                        f"logits_epoch_{epoch}": train_logits,
                        "gold": train_golds,
                       "heuristics": heuristics,
                        "original_id": original_ids,
                        "predicted_label": predictions})
  logging_dir = os.path.join(output_dir, f"training_dynamics")
  # Create directory for logging training dynamics, if it doesn't already exist.
  if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
  epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
  td_df.to_json(epoch_file_name, lines=True, orient="records")
  logger.info(f"Training Dynamics logged to {epoch_file_name}")

def log_eval_dynamics(output_dir: os.path,
                      epoch: int,
                      guids: List[int],
                      eval_logits: List[List[float]],
                      eval_golds: List[int],
                      original_ids: List[int],
                      predictions: List[int],
                      heuristics: List[List[int]],
                      ood=False):
    if ood:
        logging_dir = os.path.join(output_dir, f"eval_OOD_dynamics")
        # heuristics = [-1 for i in range(len(original_ids))]
    else:
        logging_dir = os.path.join(output_dir, f"eval_ID_dynamics")
    if len(heuristics) > 0:
        heuristics = heuristics.tolist()
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
    print(len(guids), len(eval_logits), len(eval_golds), len(original_ids), len(predictions), len(heuristics))
    record_df = pd.DataFrame({"guid": guids,
                              f"logits_epoch_{epoch}": eval_logits,
                              "gold": eval_golds,
                              "original_id": original_ids,
                              "predicted_label": predictions,
                              "heuristics": heuristics, })
    record_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"Eval Dynamics logged to {epoch_file_name}")


def read_dynamics(model_dir: os.path,
                           strip_last: bool = False,
                           id_field: str = "guid",
                           burn_out: int = None,
                  mode: str="training"):
  """
  Given path to logged training dynamics, merge stats across epochs.
  Returns:
  - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
  """
  train_dynamics = {}
  heuristics = {}
  original_idx = {}
  predicted_label = {}
  td_dir = os.path.join(model_dir, f"{mode}_dynamics")
  num_epochs = len([f for f in os.listdir(td_dir) if (os.path.isfile(os.path.join(td_dir, f)) and (f[0] !='.'))])
  if burn_out:
    num_epochs = burn_out
  logger.info(f"Reading {num_epochs} files from {td_dir} ...")
  for epoch_num in tqdm.tqdm(range(num_epochs)):
    # ### DEV ###
    # if epoch_num > 1:
    #   break
    # ### DEV ###
    epoch_file = os.path.join(td_dir, f"dynamics_epoch_{epoch_num}.jsonl")
    assert os.path.exists(epoch_file)

    with open(epoch_file, "r") as infile:
      for i, line in enumerate(infile):
        record = json.loads(line.strip())
        guid = record[id_field] if not strip_last else record[id_field][:-1]
        if guid not in train_dynamics:
          assert epoch_num == 0
          train_dynamics[guid] = {"gold": record["gold"], "logits": []}
          heuristics[guid] = {"lexical": record["heuristics"][0], "constituent": record["heuristics"][1],
                              "subsequence": record["heuristics"][2], "ood": 0}
          original_idx[guid] = record["original_id"]
          predicted_label[guid] = []

        train_dynamics[guid]["logits"].append(record[f"logits_epoch_{epoch_num}"])
        predicted_label[guid].append(record["predicted_label"])
      print('LINE NUMBER IN EPOCH {} = {}'.format(epoch_num, i))
  logger.info(f"Read training dynamics for {len(train_dynamics)} train instances.")
  return train_dynamics, heuristics, original_idx, predicted_label
