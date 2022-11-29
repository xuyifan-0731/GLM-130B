import os
import math
import json

import pdb
import torch
import types
from typing import *
from os.path import join
from bisect import bisect_right
from itertools import accumulate
from collections import defaultdict
import numpy as np
from SwissArmyTransformer import mpu
from evaluation.utils import get_tokenized_input
from scipy.stats import pearsonr, spearmanr, kendalltau

from evaluation import (
    EvaluationDataset,
    LanguageModelTask,
    LanguageModelTaskDataset,
    print_rank_0,
    LanguageModelTaskConfig,
)


def pad_batch(tokens, position_ids, attention_mask, loss_masks, targets, max_seq_length):
    attention_mask = np.pad(
        attention_mask,
        pad_width=((0, max_seq_length - len(tokens)),),
        mode="constant",
        constant_values=0,
    )
    tokens = np.concatenate((tokens, np.zeros(max_seq_length - len(tokens), dtype=np.int64)))
    targets = np.concatenate((targets, np.zeros(max_seq_length - len(targets), dtype=np.int64)))
    loss_masks = np.concatenate((loss_masks, np.zeros(max_seq_length - len(loss_masks), dtype=np.int64)))
    position_ids = np.concatenate((position_ids, np.zeros(max_seq_length - len(position_ids), dtype=np.int64)))
    return tokens, position_ids, attention_mask, loss_masks, targets


class BartscoreDataset(EvaluationDataset):
    config: LanguageModelTaskConfig

    def process_single_item(self, item):
        text, targets = get_tokenized_input(item, "inputs"), get_tokenized_input(item, "targets")
        if len(text) + self.config.generation_length + 2 > self.config.max_seq_length:
            text_length = self.config.max_seq_length - self.config.generation_length - 2
            text = text[len(text) - text_length : len(text)]
        return {"text": text, "targets": targets, "info": item.get("meta_info")}

    def __getitem__(self, idx):
        tokens = self.data[idx]["text"]
        targets = self.data[idx]["targets"]
        if len(np.array(targets).shape) > 1:
            targets = targets[0]

        mask_id = self.gmask_id if self.config.use_task_mask else self.mask_id
        sop_id = self.tokenizer.get_command("sop")

        if self.config.unidirectional:
            prompt, text = [], tokens
        else:
            prompt_length = self.config.max_seq_length - 1 - self.config.generation_length
            prompt, text = tokens[:prompt_length], targets

        seq_length = len(prompt) + len(text) + 1
        attention_mask = np.tril(np.ones((seq_length, seq_length), dtype=np.int64))
        attention_mask[: len(prompt) + 1, : len(prompt) + 1] = 1
        # if mpu.get_model_parallel_rank() == 0:
        # print(idx)
        return {
            "tokens": np.array(prompt + [mask_id, sop_id] + text[:-1], dtype=np.int64),
            "targets": np.array(prompt + [mask_id] + text, dtype=np.int64),
            "position_ids": np.arange(0, seq_length, dtype=np.int64),
            "attention_mask": attention_mask < 0.5,
            "loss_masks": np.array([0] * (len(prompt) + 1) + [1] * len(text), dtype=np.int64),
        }

    @property
    def has_collate_fn(self) -> bool:
        return True

    def collate_fn(self, samples):
        TILE = 32
        length_to_pad = (max(map(lambda spl: len(spl["tokens"]), samples)) + TILE - 1) // TILE * TILE
        token_batch, position_id_batch, attention_mask_batch = [], [], []
        loss_masks_batch, targets_batch = [], []

        for sample in samples:
            token, position_id, attention_mask, loss_masks, target = pad_batch(
                sample["tokens"],
                sample["position_ids"],
                sample["attention_mask"],
                sample["loss_masks"],
                sample["targets"],
                length_to_pad,
            )

            token_batch.append(token)
            targets_batch.append(target)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask > 0.5)
            loss_masks_batch.append(loss_masks)
        return {
            "tokens": torch.tensor(np.array(token_batch), dtype=torch.int64),
            "position_ids": torch.tensor(np.array(position_id_batch), dtype=torch.int64),
            "attention_mask": torch.tensor(np.array(attention_mask_batch), dtype=torch.int64),
            "loss_masks": torch.tensor(np.array(loss_masks_batch), dtype=torch.int64),
            "targets": torch.tensor(np.array(targets_batch), dtype=torch.int64),
        }


def calculate_bart_score(self, batch) -> List[float]:
    tokens, position_ids, attention_mask = self.process_data(batch)
    targets, loss_masks = (
        batch["targets"].to(device=torch.cuda.current_device()).long(),
        batch["loss_masks"].to(device=torch.cuda.current_device()).long(),
    )

    original_parallel_output = self.model.transformer.parallel_output
    self.model.transformer.parallel_output = False
    self.model.eval()

    with torch.no_grad():

        logits, *output_per_layers = self.model(tokens, position_ids, attention_mask, log_attention_weights=None)
        loss_fct = torch.nn.NLLLoss(reduction="none")
        lsm = torch.nn.LogSoftmax(dim=1)
        bartscore = []
        for logit, target, loss_mask in zip(logits, targets, loss_masks):
            tgt_len = sum(loss_mask.squeeze()).item()
            loss = loss_fct(lsm(logit.float().squeeze()), target.view(-1))
            loss = loss * loss_mask
            loss = loss.sum(dim=0) / tgt_len
            bartscore.append([-loss.item()])

    self.model.transformer.parallel_output = original_parallel_output
    return bartscore


class BartscoreTask(LanguageModelTask):
    def __init__(self, model, tokenizer, config):
        super().__init__(model, tokenizer, config)
        self.model.calculate_bart_score = types.MethodType(calculate_bart_score, self.model)

    @property
    def metrics(self) -> Dict[str, Callable]:
        return {"bart-score-evaluation": self.bart_score_metric}

    def bart_score_metric(self, predictions, examples):
        meta_evaluation_method = examples[0].get("info").get("meta-evaluation-method")
        human_metrics = defaultdict(list)
        human_metrics_result = defaultdict(float)
        bart_score_metric = []
        for prediction, example in zip(predictions, examples):
            if example.get("info").get("is_ground_truth") == 1:
                continue
            else:
                bart_score_metric.append(prediction)
                for metric in example.get("info").get("ground_truth_label"):
                    human_metrics[metric].append(example.get("info").get(metric))
        if meta_evaluation_method == "spearman":
            for key, values in human_metrics.items():
                spearman, _ = spearmanr(human_metrics[key], bart_score_metric)
                human_metrics_result[key] = spearman
        elif meta_evaluation_method == "pearson":
            for key, values in human_metrics.items():
                pearson, _ = pearsonr(human_metrics[key], bart_score_metric)
                human_metrics_result[key] = pearson[0]
        return human_metrics_result

    def build_dataset(self, relative_path):
        return BartscoreDataset(join(self.config.path, relative_path), self.config)

    def report_single_metrics(self, file: str, result_dict: Dict[str, float]):
        pass

    def report_group_metrics(
        self,
        group_name,
        result_dict_group: Dict[str, Tuple[Dict[str, Dict[str, float]], int]],
        level=1,
    ):
        output_str = f"    Finish group {group_name}:\n"
        print_rank_0(output_str)
        for key, values in result_dict_group.items():
            print_rank_0("file name:{fname}".format(fname=key))
            for metric_name, human_metrics in values[0].items():
                print_rank_0("metrics:{metrics}".format(metrics=metric_name))
                for human_metric, value in human_metrics.items():
                    print_rank_0(
                        "cat:{metrics}  value:{value}".format(metrics=human_metric, value=round(value * 100, 2))
                    )

    def report_overall_metrics(self, result_dict_all: Dict[str, Tuple[Dict[str, float], int]]):
        pass

    def predict_single_batch(self, batch) -> List[float]:
        return self.model.calculate_bart_score(batch)

    def sort_dict(self, dict):
        result_dict = {}
        temp_list = list(dict.keys())
        temp_list.sort()
        for item in temp_list:
            result_dict[item] = dict[item]
        return result_dict

    def kendall(self, hyp1_scores: list, hyp2_scores: list):
        """Computes the official WMT19 shared task Kendall correlation score."""
        assert len(hyp1_scores) == len(hyp2_scores)
        conc, disc = 0, 0

        for x1, x2 in zip(hyp1_scores, hyp2_scores):
            if x1 > x2:
                conc += 1
            else:
                disc += 1
        return (conc - disc) / (conc + disc)


class BartscoreWMTTask(BartscoreTask):
    def bart_score_metric(self, predictions, examples):
        better_result = defaultdict(float)
        worse_result = defaultdict(float)
        for prediction, example in zip(predictions, examples):
            if example.get("info").get("label") == "better":
                better_result[example.get("info").get("id")] = (
                    better_result[example.get("info").get("id")] + prediction / 2
                )
            else:
                worse_result[example.get("info").get("id")] = (
                    worse_result[example.get("info").get("id")] + prediction / 2
                )
        better_score = []
        worse_score = []
        better_result = self.sort_dict(better_result)
        worse_result = self.sort_dict(worse_result)
        for key, value in better_result.items():
            better_score.append(value)
        for key, value in worse_result.items():
            worse_score.append(value)
        ktau = self.kendall(better_score, worse_score)
        return {"kendall": ktau}


class BartscoreD2TTask(BartscoreTask):
    def bart_score_metric(self, predictions, examples):
        result_0 = defaultdict(lambda: -100.0)
        result_1 = defaultdict(lambda: -100.0)
        human_metrics = defaultdict(float)
        for prediction, example in zip(predictions, examples):
            if isinstance(prediction, list):
                prediction = prediction[0]

            human_metrics[example.get("info").get("id")] = example.get("info").get("informativeness")
            if example.get("info").get("reverse") == 0:
                result_0[example.get("info").get("id")] = max(result_0[example.get("info").get("id")], prediction)
            else:
                result_1[example.get("info").get("id")] = max(result_1[example.get("info").get("id")], prediction)
        result_0_score = []
        result_1_score = []
        result_0 = self.sort_dict(result_0)
        result_1 = self.sort_dict(result_1)
        human_metrics = self.sort_dict(human_metrics)
        f1_result = defaultdict(float)
        for key, value in result_0.items():
            result_0_score.append(value)
            f1_result[key] = f1_result[key] + value / 2
        for key, value in result_1.items():
            result_1_score.append(value)
            f1_result[key] = f1_result[key] + value / 2
        ktau = self.kendall(result_0_score, result_1_score)
        f1_result = self.sort_dict(f1_result)
        spearman, _ = spearmanr(list(f1_result.values()), list(human_metrics.values()))

        return {"kendall": ktau, "spearman": spearman}


class BartscoreR19Task(BartscoreTask):
    def bart_score_metric(self, predictions, examples):
        better_result = defaultdict(float)
        worse_result = defaultdict(float)
        for prediction, example in zip(predictions, examples):
            if example.get("info").get("label") == "correct":
                better_result[example.get("info").get("id")] = prediction
            else:
                worse_result[example.get("info").get("id")] = prediction
        better_score = []
        worse_score = []
        better_result = self.sort_dict(better_result)
        worse_result = self.sort_dict(worse_result)
        for key, value in better_result.items():
            better_score.append(value)
        for key, value in worse_result.items():
            worse_score.append(value)
        acc = sum([x > y for x, y in zip(better_score, worse_score)]) / len(worse_score)
        return {"acc": acc}
