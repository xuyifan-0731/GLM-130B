import os
import math
import json

import pdb
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
        if mpu.get_model_parallel_rank() == 0:
            print(idx)
        return {
            "tokens": np.array(prompt + [mask_id, sop_id] + text[:-1], dtype=np.int64),
            "targets": np.array(prompt + [mask_id] + text, dtype=np.int64),
            "position_ids": np.arange(0, seq_length, dtype=np.int64),
            "attention_mask": attention_mask < 0.5,
            "loss_masks": np.array([0] * (len(prompt) + 1) + [1] * len(text), dtype=np.int64),
        }


class BartscoreTask(LanguageModelTask):
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
                # human_metrics.append(example.get("info").get("human_score"))
        if meta_evaluation_method == "spearman":
            for key, values in human_metrics.items():
                spearman, _ = spearmanr(human_metrics[key], bart_score_metric)
                human_metrics_result[key] = spearman
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
