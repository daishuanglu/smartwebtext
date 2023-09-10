import pandas as pd
from utils.string_utils import damerauLevenshtein


def label2searchkw(ml_col, gt_col):
    label_dict={}
    for col in ml_col:
        max_sim= 0
        for c in gt_col:
            sim = damerauLevenshtein(col,c, similarity=True)
            if sim>max_sim:
                max_sim = sim
                label_dict[c]=col
    return label_dict


def thresholded_precision(prediction_file, kws, th=0.5):
    pred_df = pd.read_csv(prediction_file)
    pred_cols = [col.split(':') for col in pred_df.columns if (':' in col) and ('haskey' not in col)]
    method, pred_words = zip(*pred_cols)
    method = list(method)[0]
    pred_words = set(pred_words)
    label = label2searchkw(pred_words, kws)
    results = {w: {} for w in kws}
    for w in kws:
        eval_col = ':'.join([method, label[w]])
        pos_recall = (pred_df[eval_col]> th ).mean()
        print("method %s, thresholded at %.2f, positive recall for keyword '%s'=%.2f" % (
            method, th, w, pos_recall
        ))
        results[w][method] = pos_recall
    return results