from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
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


def score(predictions: pd.Series, groundtruths):
    groundtruths = groundtruths.dropna()
    # Find missing indices in series2 compared to series1
    #missing_indices = groundtruths.index.difference(predictions.index)
    # Add missing indices to series2 with NaN values
    predictions = predictions.reindex(groundtruths.index)
    predictions = predictions.loc[groundtruths.index]
    n_samples = len(predictions)
    print('%d/%d missing predictions.' % (predictions.isna().sum(), n_samples))
    thresholds = np.linspace(0, 1, num=101)
    df_prec_rec = defaultdict(list)
    for th in thresholds:
        df_prec_rec['precision'].append(precision_score(groundtruths == 1.0, predictions > th))
        df_prec_rec['recall'].append(recall_score(groundtruths == 1.0, predictions > th))
        df_prec_rec['threshold'].append(th)
        df_prec_rec['f_score'].append(f1_score(groundtruths == 1.0, predictions > th))
    df_prec_rec = pd.DataFrame.from_dict(df_prec_rec)
    df_prec_rec = df_prec_rec.sort_values(by=['precision', 'recall'])
    df_prec_rec = df_prec_rec.drop_duplicates(subset = ['recall'], keep = 'last')
    df_prec_rec = df_prec_rec.sort_values(by='recall')
    invalid = (df_prec_rec['precision'] == 0) & (df_prec_rec['recall'] == 0)
    df_prec_rec = df_prec_rec[~invalid]
    default_zero_recall = {'precision': 1.0, 'recall': 0.0, 'threshold': 1.0, 'f_score': 1.0}
    default_100_recall = {'precision': 0.0, 'recall': 1.0, 'threshold': -1.0, 'f_score': 0.0}
    df_prec_rec = pd.concat([
        pd.DataFrame([default_zero_recall], columns=df_prec_rec.columns),
        df_prec_rec,
        pd.DataFrame([default_100_recall], columns=df_prec_rec.columns),], ignore_index=True)
    return df_prec_rec


def auc_precision_recall(precision, recall):
    # Ensure that precision and recall lists are of the same length
    if len(precision) != len(recall):
        raise ValueError("Precision and recall lists must be of the same length")
    # Sort the recall and precision pairs by recall in ascending order
    recall, precision = zip(*sorted(zip(recall, precision)))
    auc = 0.0
    for i in range(1, len(precision)):
        auc += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2.0
    return auc


def load_predictions_df(prediction_files, index_key, header_mappings=[]):
    merged_data = None
    for file, header_mapping in zip(prediction_files, header_mappings):
        df = pd.read_csv(file, parse_dates=False, keep_default_na=False, na_values=[])
        rename_dict = {k: v for k,v in header_mapping.items() if k in df.columns}
        df = df.rename(rename_dict, axis=1)
        df = df.set_index(index_key)
        df = df[[col for col in df.columns if ':' in col]]
        if merged_data is None:
            merged_data = df
        else:
            merged_data = pd.merge(
                merged_data, df, left_index=True, right_index=True, how='outer')
    return merged_data


def load_gt(pro_label_file, index_key):
    df_gt = pd.read_csv(
        pro_label_file, dtype=str, parse_dates=False, na_values=[], keep_default_na=False)
    print('%d groundtruth company labels loaded. ' % len(df_gt))
    df_gt = df_gt.set_index(index_key)
    return df_gt