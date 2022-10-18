from string import punctuation
from functools import partial
from typing import List
from SwissArmyTransformer import mpu
import numpy as np
import torch

from evaluation import qa_evaluate, GenerationTask
from collections import defaultdict
from typing import Dict,Tuple

from evaluation.utils import (
    print_rank_0,
    get_tokenized_input,
)


def exact_match_score(prediction, ground_truth):
    return prediction.strip() == ground_truth.strip()


class PILE(GenerationTask):
    def __init__(self, model, tokenizer, config_path):
        super(PILE, self).__init__(model, tokenizer, config_path)

    def cal_n_gram(self,input_list, ngram_num):
        if len(input_list) <= ngram_num:
            return 0
        else:
            total_gram = len(input_list) - ngram_num + 1
            no_repeat_gram = len(set(zip(*[input_list[i:] for i in range(ngram_num)])))
            return 100*(1-no_repeat_gram/total_gram)

    # def cal_mauve(self, predictions, examples):


    def PileMetric(self, predictions, examples):
        metrics_dict = defaultdict(lambda: [])
        for text in predictions:           
            if torch.is_tensor(text):
                text = text.tolist()
            #print(text)
            rep_2 = self.cal_n_gram(text, 2)
            rep_3 = self.cal_n_gram(text, 3)
            rep_4 = self.cal_n_gram(text, 4)
            diversity = (1-rep_2/100)*(1-rep_3/100)*(1-rep_4/100)
            #print(rep_2,rep_3,rep_4,diversity)
            metrics_dict["rep_2"].append(rep_2)
            metrics_dict["rep_3"].append(rep_3)
            metrics_dict["rep_4"].append(rep_4)
            metrics_dict["diversity"].append(diversity)
        return metrics_dict

    @property
    def metrics(self):
        return {"pile": self.PileMetric}

    def predict_single_batch(self, batch) -> List[List[int]]:
        output = self.model.generate_text(batch, self.strategy, return_all_beams=False)
        return output

    def report_single_metrics(self, file: str, result_dict: Dict[str, float]):
        pass

    def report_group_metrics(self, group_name, result_dict_group: Dict[str, Tuple[Dict[str, float], int]], level=1):
        print("report")
        for tmp1 in result_dict_group.values():
            tmp1 = tmp1[0]
            for result in tmp1.values():
                for key,values in result.items():
                    print_rank_0(key,np.mean(values))

                

