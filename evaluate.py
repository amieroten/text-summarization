'''
Module to evaluate auto-summarization output,
using ROUGE.
'''
from pathlib import Path
import glob

import numpy as np

from rouge_score import rouge_scorer

def evaluate_one(model: str, index):
    gold_path = Path("output/gold/")

def evaluate_all(model_type: str):
    gold_dir = "output/" + model_type + "/gold/"
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL','rougeLsum'])
    all_scores = {'rouge1': [],'rouge2': [],'rougeL': [] ,'rougeLsum': []}

    for file in glob.glob(gold_dir +"*.txt"):
        gold = " ".join(open(file,'r').readlines())
        model = " ".join(open(file.replace("gold","model"),"r").readlines())
        scores = scorer.score(gold, model)
        for type, score in scores.items():
            all_scores[type].append(score)

    all_averages = {}
    for type in all_scores.keys():
        all_averages[type] = {}
        all_averages[type]["f_mean"] = np.mean([score.fmeasure for score in all_scores[type]])
        all_averages[type]["f_sd"] = np.std([score.fmeasure for score in all_scores[type]])
        all_averages[type]["recall_mean"] = np.mean([score.recall for score in all_scores[type]])
        all_averages[type]["recall_sd"] = np.std([score.recall for score in all_scores[type]])
        all_averages[type]["precision_mean"] = np.mean([score.precision for score in all_scores[type]])
        all_averages[type]["precision_sd"] = np.std([score.precision for score in all_scores[type]])
    print(all_averages)
    return all_scores, all_averages

if __name__ == "__main__":
    all_scores, all_averages = evaluate_all("LEAD-3_val")
    print("pause")

