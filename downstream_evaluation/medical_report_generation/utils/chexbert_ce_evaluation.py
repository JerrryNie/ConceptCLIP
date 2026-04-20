from pprint import pprint
import pandas as pd
import os
from chexbert import CheXbert
import pandas as pd
import numpy as np
#from torchmetrics import Metric
from sklearn.metrics import precision_recall_fscore_support

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]

class CheXbertMetrics():
    def __init__(self, checkpoint_path, mbatch_size, device):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.chexbert = CheXbert(self.checkpoint_path, self.device,).to(self.device)

    def mini_batch(self, gts, res, mbatch_size=16):
        length = len(gts)
        assert length == len(res)
        for i in range(0, length, mbatch_size):
            yield gts[i:min(i + mbatch_size, length)], res[i:min(i + mbatch_size, length)]

    def compute(self, gts, res):
        gts_chexbert = []
        res_chexbert = []
        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            gt_chexbert = self.chexbert(list(gt)).tolist()
            re_chexbert = self.chexbert(list(re)).tolist()
            gts_chexbert += gt_chexbert
            res_chexbert += re_chexbert
        gts_chexbert = np.array(gts_chexbert)
        res_chexbert = np.array(res_chexbert)
        
        res_chexbert_cvt = (res_chexbert == 1)
        gts_chexbert_cvt = (gts_chexbert == 1)

        tp = (res_chexbert_cvt * gts_chexbert_cvt).astype(float)

        fp = (res_chexbert_cvt * ~gts_chexbert_cvt).astype(float)
        fn = (~res_chexbert_cvt * gts_chexbert_cvt).astype(float)

        tp_cls = tp.sum(0)
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)

        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        precision_class = np.nan_to_num(tp_cls / (tp_cls + fp_cls + 1e-6))
        recall_class = np.nan_to_num(tp_cls / (tp_cls + fn_cls + 1e-6))
        f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls) + 1e-6))

        scores_cvt = {
            'ce_precision_macro': precision_class.mean(),
            'ce_recall_macro': recall_class.mean(),
            'ce_f1_macro': f1_class.mean(),
            'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum() + 1e-6),
            'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum() + 1e-6),
            'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum()) + 1e-6),
            'ce_precision_example': np.nan_to_num(tp_eg / (tp_eg + fp_eg + 1e-6)).mean(),
            'ce_recall_example': np.nan_to_num(tp_eg / (tp_eg + fn_eg + 1e-6)).mean(),
            'ce_f1_example': np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg) + 1e-6)).mean(),
            'ce_num_examples': float(len(res_chexbert)),
        } 
        ###################################################
        return scores_cvt, f1_class



def main():
    chexbert_metrics = CheXbertMetrics('chexbert.pth', 16, 'cpu') #NOTE: change with your chexbert model path
    report_path = "../results/conceptclip_l_iu/conceptclip_l_iu.csv"
    report_data = pd.read_csv(report_path)
    
    gt_report_column = 'GT_whole_report'
    pred_report_column = 'Pred_whole_report'

    gt_list = report_data[gt_report_column].tolist()
    res_list = report_data[pred_report_column].tolist()

    ce_scores, f1_class = chexbert_metrics.compute(gt_list, res_list)
    for k, v in ce_scores.items():
        print(f"{k}: {v}")
        
    # save the results to a csv file
    ce_scores_df = pd.DataFrame(ce_scores, index=[0])
    save_dir = os.path.dirname(report_path)
    save_file = os.path.join(save_dir, "ce_evaluation_results.csv")
    ce_scores_df.to_csv(save_file, index=False)


if __name__ == '__main__':
    main()
