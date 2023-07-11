
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from typing import Dict, Any
from fastDamerauLevenshtein import damerauLevenshtein
import pandas as pd
from tabulate import tabulate


f=open("evaluators/prnews/MTurk/MTurkTemplate.txt",'r')
template=f.readlines()
f.close()

f=open("evaluators/prnews/MTurk/MTurkHead.txt",'r')
head=f.readlines()
f.close()

f=open("evaluators/prnews/MTurk/MTurkTail.txt",'r')
tail=f.readlines()
f.close()

df_batch_tic = pd.read_csv('evaluators/prnews/MTurk/MTurk_eval_tics.csv')
df_batch_tic = df_batch_tic[['tic', 'company']]

eval_terms = ["analytics","innovation","technology"]

def render_mturk_html(df):
    val_id = 1
    box_id = 0

    def get_block(i, val_id,box_id):
        block = []
        for l in template:
            if "<!--company-->" in l:
                l = l.replace("<!--company-->", df['company'][i].upper())
            if "<!--box-->" in l:
                l = l.replace("<!--box-->", str(box_id))
                if "input" in l:
                    box_id += 1
            if "<!--link-->" in l:
                l = l.replace("<!--link-->", df['link'][i])
            if "<!--value-->" in l:
                l = l.replace("<!--value-->", str(val_id))
                if "input" in l:
                    val_id += 1
            if "<!--Qnum-->" in l:
                l = l.replace("<!--Qnum-->", '{:d}/{:d}'.format(i + 1, len(df)))
            block.append(l)

        return block,val_id,box_id

    blocks=[]
    for i in range(len(df)):
        block,val_id,box_id=get_block(i,val_id,box_id)
        blocks += block

    fo=open("evaluators/prnews/MTurk/renderedTurk.txt",'w')
    fo.writelines(head+blocks+tail)
    fo.close()


def append_MTurk_batch_response(batch_file, nvoters=None):
    batch=pd.read_csv(batch_file)
    res_col_names=[col for col in batch.columns if 'checkboxes' in col]
    batch=batch[res_col_names]
    res_col_names_={c:int(c.split('.')[-1]) for c in res_col_names}
    batch=batch.rename(res_col_names_,axis=1)
    batch=batch.sort_index(axis=1)
    #print(batch)
    #values=batch.values[:5,:].T
    if nvoters is None:
        nvoters = len(batch)
    values = batch.values[:nvoters,:].T
    eval_term_id = np.reshape(batch.columns.tolist(),(-1, len(eval_terms)))
    headers=[o+'_'+str(i)  for o in eval_terms for i in range(nvoters)]
    values_matrix=[]
    for i,term in enumerate(eval_terms):
        values_matrix.append(values[eval_term_id[:,i]-1,:])
    values_matrix = np.hstack(values_matrix)
    df_vote=pd.DataFrame(values_matrix,columns=headers)
    return pd.concat([df_batch_tic,df_vote], axis=1)


def majority_vote(df):
    nvote= {}
    maj_vote = defaultdict()
    for col in df.columns:
        if '_' in col:
            name,_ = col.split('_')
            if name in nvote.keys():
                nvote[name]+=1
            else:
                nvote[name] =1
    for col_name, cnt in nvote.items():
        # if the same vote then YES
        voter_cols = [col_name + '_' + str(i) for i in range(nvote[col_name])]
        maj_vote[col_name] = df[voter_cols].sum(1) >= (cnt//2-1)
        #maj_vote[col_name] = df[voter_cols].sum(1) >= 1
        maj_vote[col_name] = maj_vote[col_name].astype(int)
    for col in ['company', 'tic']:
        maj_vote[col] = df[col]
    return pd.DataFrame(maj_vote)


def join_pred_gt(df_pred, df_gt):
    df_predict = pd.merge(df_gt, df_pred , how='left', left_index=True, right_index=True)
    df_gt_local = df_predict[df_gt.columns]
    return df_predict, df_gt_local


def _print_df(df):
    tbl = tabulate(df, headers='keys', tablefmt='psql')
    print(tbl)
    return


def label2searchkw(ml_col, gt_col):
    label_dict={}
    for col in ml_col:
        max_sim= 0
        for c in gt_col:
            sim = damerauLevenshtein(col,c, similarity=True)
            if sim>max_sim:
                max_sim = sim
                label_dict[col]=c
    return label_dict

def process_predictions_df(df, kws):
    methods, ml_col_names = zip(*[c.split(':') for c in df.columns if ':' in c])
    methods = set(methods)-{'haskey'}
    ml_col_names = sorted(set(ml_col_names))
    kw_col_map = label2searchkw(ml_col_names, kws)
    results = {}
    for method in methods:
        df_results = {method+':'+kw_col_map[col]: df[method+':'+col].tolist()\
                for col in ml_col_names}
        #df_results['link'] = df['link'].tolist()
        df_results['tic'] = df['tic'].tolist()
        #df_results.update({'haskey:' + kw_col_map[col]: df['haskey:'+col].tolist() \
        #              for col in ml_col_names})
        results[method] = pd.DataFrame.from_dict( df_results)
        results[method] = results[method].drop_duplicates('tic')
        results[method] = results[method].set_index('tic')
    return results

colormaps=['orange','purple','blue','red','green','yellow','black','brown']


def eval_plot(df_pred: Dict[str, Any], df_gt: pd.DataFrame, kword):
    segments = {
        'full': (lambda x: x.index!=None),
        'haskey': (lambda x: (x['haskey:'+kword]>0)),
        'nothaskey': (lambda x: ~(x['haskey:'+kword]>0))
    }
    fig, axes = plt.subplots(1, len(segments))

    def _plot(ax, pred ,gt, c, l, coverage):
        precision, recall, thresholds = precision_recall_curve(
            gt, pred, pos_label=1)
        recall= coverage * recall
        map = average_precision_score(
            gt, pred, average='micro', pos_label=1)
        axes[i].plot(recall, precision, color=c, label=l)
        leg = l + '- mAP:' + str(map)
        return ax, leg

    for i,(seg, ind_fn) in enumerate(segments.items()):
        legend=[]
        seg_ind = {}
        for im, method in enumerate(df_pred.keys()):
            # remove companies that do not predicted OR do not have GT labels
            df_predict, df_gt_local = join_pred_gt(df_pred[method], df_gt)
            #coverage = len(df_predict)/len(df_gt.drop_duplicates('tic'))
            print("METHOD=%s, %d companies can be evaluated after dropping duplicated tic." % (method, len(df_predict)) )
            ind = ind_fn(df_predict)
            for ii, t in enumerate(df_predict.index):
                seg_ind[t] = ind.tolist()[ii]
            axes[i], leg= _plot(
                axes[i],
                df_predict[method+':'+kword][ind],
                df_gt_local[kword][ind],
                colormaps[im], method, coverage=1.0)
            legend.append(leg)
        axes[i].legend(legend, loc='best')
        seg_ratio = np.mean(list(seg_ind.values()))
        axes[i].set_title('%s:ROC - %s, ratio: %.2f' % (kword, seg, seg_ratio) )
        axes[i].set_ylabel('precision')
        axes[i].set_xlabel('recall')
    #plt.show(block=True)
    return fig, axes


def cf_haskey(eval_data_path, context_col, ref_col, eval_refs, query_kws):
    df = pd.read_csv(eval_data_path)
    df = df[df[ref_col].isin(eval_refs)]
    df = df[[context_col, ref_col]]
    haskey = defaultdict(list)
    for ref in sorted(set(df[ref_col])):
        haskey[ref_col].append(ref)
    for kw in query_kws:
        for ref in haskey[ref_col]:
            ind = (ref == df[ref_col])
            texts = ' '.join(set(df[ind][context_col]))
            haskey['haskey:' + kw].append(kw in texts)
    return pd.DataFrame.from_dict(haskey)



def score(predictions, groundtruths):
    groundtruths = groundtruths.dropna()
    # Find missing indices in series2 compared to series1
    missing_indices = groundtruths.index.difference(predictions.index)
    # Add missing indices to series2 with NaN values
    predictions = predictions.append(pd.Series(index=missing_indices, dtype='float'))
    predictions = predictions.loc[groundtruths.index]
    n_samples = len(predictions)
    print('%d/%d missing predictions.' % (predictions.isna().sum(), n_samples))
    thresholds = np.linspace(0, 1, num=101)
    df_prec_rec = defaultdict(list)
    for th in thresholds:
        df_prec_rec['precision'].append(precision_score(groundtruths == 1.0, predictions > th))
        df_prec_rec['recall'].append(recall_score(groundtruths == 1.0, predictions > th))
        df_prec_rec['threshold'].append(th)
    df_prec_rec = pd.DataFrame.from_dict(df_prec_rec)
    df_prec_rec = df_prec_rec.sort_values(by=['precision', 'recall'])
    #idx = df_prec_rec.groupby('precision')['recall'].idxmax()
    df_prec_rec = df_prec_rec.drop_duplicates(subset = ['recall'], keep = 'last')
    df_prec_rec = df_prec_rec.sort_values(by='recall')
    invalid = (df_prec_rec['precision'] == 0) & (df_prec_rec['recall'] == 0)
    df_prec_rec = df_prec_rec[~invalid]
    default_zero_recall = {'precision': 1.0, 'recall': 0.0, 'threshold': 1.0}
    default_100_recall = {'precision': 0.0, 'recall': 1.0, 'threshold': -1.0}
    df_prec_rec = pd.DataFrame(
        [default_zero_recall], columns=df_prec_rec.columns).append(df_prec_rec, ignore_index=True)
    df_prec_rec = df_prec_rec.append(default_100_recall, ignore_index=True)
    return df_prec_rec


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


def load_MTurk(index_key):
    batch_file = "evaluators/prnews/MTurk/Batch_4766827_batch_results.csv"
    df_vote = append_MTurk_batch_response(batch_file)
    df_maj_vote = majority_vote(df_vote)
    df_maj_vote = df_maj_vote.drop_duplicates('tic')
    nsamples = len(df_maj_vote)
    print("MTurk voted on %d companies." % nsamples)
    df_maj_vote.to_csv("evaluators/prnews/MTurk/MTurk_maj_vote.csv", index=False)
    df_maj_vote= df_maj_vote.set_index(index_key)
    return df_maj_vote


if __name__=="__main__":
    """
    render_mturk_html(df)
    """
    unused_methods = ['haskey', 'edit_sim_sent']
    index_key = 'company'
    kws = ['analytics', 'innovation', 'technology']
    prediction_files = [
        #'evaluation/prnews_accounting/local_topics_eval_predictions.csv',
        'evaluation/prnews_accounting/prnews_global_topic_emb_val_predictions.csv',
        'evaluation/prnews_accounting/prnews_edit_val_predictions.csv',
        #'evaluation/prnews_accounting/tte_sent_small_eval_predictions.csv'
    ]
    header_mappings = [
        #{'local_topics_sim:analyt': 'local_topics_sim:analytics',
        # 'local_topics_sim:innov':	'local_topics_sim:innovation',
        # 'local_topics_sim:technolog': 'local_topics_sim:technology'},
        {'global_topics_sim:analyt': 'global_topics_sim:analytics',
         'global_topics_sim:innov': 'global_topics_sim:innovation',
         'global_topics_sim:technolog': 'global_topics_sim:technology'},
        {'edit_sim:analytic': 'edit_sim:analytics', 'haskey:analytic': 'haskey:analytics'},
        #{'Company':'company'}
    ]
    #df_gt = load_gt(GT_CSV, index_key)
    df_gt = load_MTurk(index_key)
    df_predictions = load_predictions_df(prediction_files, index_key, header_mappings)
    methods = sorted(set([col.split(':')[0] for col in df_predictions.columns]))

    for kw in kws:
        fig, (ax, ax1, ax2) = plt.subplots(1, 3)
        i=0
        lengs = []
        for method in methods:
            if method in unused_methods:
                continue
            score_col = ':'.join([method, kw])
            lengs.append(score_col)
            print(' ----------------- Threshold selector for ', score_col, ' ------------------')
            df_prec_rec = score(df_predictions[score_col], df_gt[kw])
            _print_df(df_prec_rec)
            ax.plot(df_prec_rec['recall'], df_prec_rec['precision'], '.--', c=colormaps[i])
            i+=1
        ax.legend(lengs, loc='best')
        ax.set_title('%s:ROC' % kw)
        ax.set_ylabel('precision')
        ax.set_xlabel('recall')
        haskey = (df_predictions['haskey:%s' % kw] == 1.0)
        df_predictions_haskey = df_predictions[haskey]
        common_idx = df_predictions_haskey.index.intersection(df_gt.index)
        df_gt_haskey = df_gt.loc[common_idx]
        i = 0
        for method in methods:
            if method in unused_methods:
                continue
            score_col = ':'.join([method, kw])
            lengs.append(score_col)
            print(' ----------------- haskey Threshold selector for ', score_col, ' ------------------')
            df_prec_rec = score(df_predictions_haskey[score_col], df_gt_haskey[kw])
            _print_df(df_prec_rec)
            ax1.plot(df_prec_rec['recall'], df_prec_rec['precision'], '.--', c=colormaps[i])
            i+=1
        ax1.legend(lengs, loc='best')
        ax1.set_title('%s_haskey:ROC' % kw)
        ax1.set_ylabel('precision')
        ax1.set_xlabel('recall')
        df_predictions_nothaskey = df_predictions[~haskey]
        common_idx = df_predictions_nothaskey.index.intersection(df_gt.index)
        df_gt_nothaskey = df_gt.loc[common_idx]
        i = 0
        for method in methods:
            if method in unused_methods:
                continue
            score_col = ':'.join([method, kw])
            lengs.append(score_col)
            print(' ----------------- Nothaskey Threshold selector for ', score_col, ' ------------------')
            df_prec_rec = score(df_predictions_nothaskey[score_col], df_gt_nothaskey[kw])
            _print_df(df_prec_rec)
            ax2.plot(df_prec_rec['recall'], df_prec_rec['precision'], '.--', c=colormaps[i])
            i+=1
        ax2.legend(lengs, loc='best')
        ax2.set_title('%s_nothaskey:ROC' % kw)
        ax2.set_ylabel('precision')
        ax2.set_xlabel('recall')
        plt.show(block=True)
