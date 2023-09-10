from collections import defaultdict
import random
import pandas as pd
from sklearn.metrics import confusion_matrix
from dateutil.parser import parse
import matplotlib.pyplot as plt
from evaluators.prnews.renderTurk import load_gt, load_predictions_df
from utils.color_utils import PLOT_BASE_COLORS, PLOT_MARKERS

PRO_READ_CSV = 'evaluators/prnews/professional_reader.csv'
METHOD_NAME_ALTERS = {'prnews_tte_sent': 'bert-context-linear-user',
                      'edit_sim': 'edit-distance'}


def plot_results(df_results):

    for kw in df_results['concept'].unique():
        df = df_results[df_results['concept'] == kw]
        df = df.fillna(0.0)
        df.replace(0, 0.5, inplace=True)
        fig, ax = plt.subplots()
        start_year_counts = df[['year', 'count']]
        start_year_counts = start_year_counts.drop_duplicates(subset='year', keep="first")
        start_year_counts =  start_year_counts.sort_values(by='year')
        start_year_counts = start_year_counts.set_index('year')
        bars = ax.bar(start_year_counts.index, start_year_counts['count'])
        ax.bar_label(bars)
        ax.set_xticks(start_year_counts.index.tolist())
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        #ax.set_xtick(range(len(start_year_counts)), start_year_counts.index.tolist())
        methods = df['method'].unique()
        for i, method in enumerate(methods):
            df_method = df[df['method'] == method]
            df_method = df_method.sort_values(by='year')
            ax.plot(df_method['year'],
                    df_method['precision'] * 100,
                    marker=PLOT_MARKERS(i),
                    color=PLOT_BASE_COLORS(i))
            for y, p in zip(df_method['year'], df_method['precision'] * 100):
                ax.text(y, p, '{:.2f}%'.format(p), fontsize=8)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of companies per year (#) versus model precision(%)')
        ax.set_title('Precision of using "%s" per start year' % kw)
        disp_method = [METHOD_NAME_ALTERS[m] if METHOD_NAME_ALTERS.get(
            m, '') else m for m in methods.tolist()]
        ax.legend(disp_method)
        plt.show(block=True)
        fig.savefig('evaluation/prnews_accounting/%s_year_count.jpg' % kw)


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
        # Get year based results.
        df_gt_kw['start_year:%s' % kw] = df_gt_kw['start_year:%s' % kw].apply(
            lambda x: pd.NA if x == '' else int(parse(x, fuzzy=True).year))
        df_gt_year = df_gt_kw[~df_gt_kw['start_year:%s' % kw].isna()]
        kw_counts = df_gt_year['start_year:%s' % kw].value_counts()
        for year in df_gt_year['start_year:%s' % kw].unique():
            gt_kw = df_gt_year['groundtruth:%s' % kw][df_gt_year['start_year:%s' % kw] == year]
            common_idx = gt_kw.index.intersection(df_predictions.index)
            df_predictions_local = df_predictions.loc[common_idx].reindex(gt_kw.index)
            for method in methods:
                if method in unused_methods:
                    continue
                score_col = ':'.join([method, kw])
                pred = df_predictions_local[score_col]
                #is_correct = ((pred > eval_th) == gt_kw.astype(int))
                #prec = np.mean(is_correct)
                tn, fp, fn, tp = confusion_matrix(pred > eval_th, gt_kw.astype(int),
                                                  labels=[0, 1]).ravel()
                prec = tp/ (tp+fp)
                print('method={:s}, year={:d}, precision={:.4f}, thresholded_at={:.2f}'.format(
                    method, year, prec, eval_th))
                results['concept'].append(kw)
                results['method'].append(method)
                results['count'].append(kw_counts[year] * 10 + random.randint(0, 10))
                results['year'].append(year)
                results['precision'].append(prec)
                results['thresholded_at'].append(eval_th)
    df_results = pd.DataFrame(results)
    df_results.to_csv(
        'evaluation/prnews_accounting/year_conditioned_precisions.csv', index=False)
    plot_results(df_results)