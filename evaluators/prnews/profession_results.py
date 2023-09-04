from collections import defaultdict
import numpy as np
import pandas as pd
from utils.string_utils import damerauLevenshtein
from dateutil.parser import parse
import matplotlib.pyplot as plt
from evaluators.prnews.renderTurk import load_gt, load_predictions_df


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

PRO_READ_CSV = 'evaluators/prnews/professional_reader.csv'


if __name__=="__main__":
    unused_methods = ['haskey', 'edit_sim_sent']
    kws = ['analytics']
    prediction_files = [
        'evaluation/prnews_accounting/prnews_local_topic_emb_val_predictions.csv',
        'evaluation/prnews_accounting/prnews_global_topic_emb_val_predictions.csv',
        'evaluation/prnews_accounting/prnews_edit_val_predictions.csv',
        'evaluation/prnews_accounting/prnews_tte_sent_val_predictions.csv'
    ]
    header_mappings = [
        {'local_topics_sim:analyt': 'local_topics_sim:analytics',
         'local_topics_sim:innov': 'local_topics_sim:innovation',
         'local_topics_sim:technolog': 'local_topics_sim:technology'},
        {'global_topics_sim:analyt': 'global_topics_sim:analytics',
         'global_topics_sim:innov': 'global_topics_sim:innovation',
         'global_topics_sim:technolog': 'global_topics_sim:technology'},
        {'edit_sim:analytic': 'edit_sim:analytics', 'haskey:analytic': 'haskey:analytics'},
        {}
    ]

    df_gt = load_gt(PRO_READ_CSV, index_key='company')
    df_predictions = load_predictions_df(prediction_files, 'company', header_mappings)
    methods = sorted(set([col.split(':')[0] for col in df_predictions.columns]))
    eval_th = 0.5
    results = defaultdict(list)
    for kw in kws:
        df_gt['groundtruth:%s' % kw] = pd.to_numeric(df_gt['groundtruth:%s' % kw], errors='coerce')
        df_gt_kw = df_gt[~df_gt['groundtruth:%s' % kw].isna()]
        print('load %d %s gt samples.' % (len(df_gt_kw), kw))
        fig, ax = plt.subplots(1, 1)
        # Get year based results.
        df_gt_kw['start_year:%s' % kw] = df_gt_kw['start_year:%s' % kw].apply(
            lambda x: pd.NA if x == '' else int(parse(x, fuzzy=True).year))
        df_gt_year = df_gt_kw[~df_gt_kw['start_year:%s' % kw].isna()]
        year_counts = df_gt_year['start_year:%s' % kw].value_counts()
        year_counts = year_counts.sort_index()
        year_counts.plot.bar()
        for i, count in enumerate(year_counts):
            ax.text(i, count, str(count), ha='center', va='bottom')
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.set_title('Number of company using "%s" start year counts' % kw)
        plt.show(block=True)
        fig.savefig('evaluation/prnews_accounting/%s_year_count.jpg' % kw)
        plt.close()
        for year in df_gt_year['start_year:%s' % kw].unique():
            gt_kw = df_gt_year['groundtruth:%s' % kw][df_gt_year['start_year:%s' % kw] == year]
            common_idx = gt_kw.index.intersection(df_predictions.index)
            df_predictions_local = df_predictions.loc[common_idx].reindex(gt_kw.index)
            for method in methods:
                if method in unused_methods:
                    continue
                score_col = ':'.join([method, kw])
                pred = df_predictions_local[score_col]
                #print(pred)
                #print(gt_kw)
                is_correct = ((pred > eval_th) == gt_kw.astype(int))
                print('method={:s}, year={:d}, precision={:.4f}, thresholded_at={:.2f}'.format(
                    method, year, np.mean(is_correct), eval_th))
                results['concept'].append(kw)
                results['method'].append(method)
                results['year'].append(year)
                results['precision'].append(np.mean(is_correct))
                results['thresholded_at'].append(eval_th)
    pd.DataFrame(results).to_csv(
        'evaluation/prnews_accounting/year_conditioned_precisions.csv', index=False)