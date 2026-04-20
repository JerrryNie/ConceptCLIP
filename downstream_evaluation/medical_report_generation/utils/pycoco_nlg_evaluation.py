from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import pandas as pd

import os


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

if __name__ == "__main__":
    results_path = "../results/conceptclip_l_iu/conceptclip_l_iu.csv"

    gt_report_column = 'GT_whole_report'
    pred_report_column = 'Pred_whole_report'
    id_column = 'Image_ID'

    df = pd.read_csv(results_path)

    # print number of samples
    print(f"Number of samples: {len(df)}")  
    
    # 定义两个空字典，用于存储提取的信息
    gt_dict = {}
    pred_dict = {}
    
    # 遍历DataFrame的每一行
    for _, row in df.iterrows():
        # 获取对应的ID、GT和Pred
        id_value = row[id_column]
        gt_value = row[gt_report_column]
        pred_value = row[pred_report_column]

        # 将ID作为键，GT和Pred作为值存储在字典中
        gt_dict[id_value] = [gt_value]
        pred_dict[id_value] = [pred_value]
        
    # 计算评价指标
    metrics = compute_scores(gt_dict, pred_dict)
    print(metrics)
    # save the results to a csv file
    metrics_df = pd.DataFrame(metrics, index=[0])
    # dir 一样，只是文件名不一样
    save_dir = os.path.dirname(results_path)
    save_file = os.path.join(save_dir, "nlg_evaluation_results.csv")
    metrics_df.to_csv(save_file, index=False)
    
    