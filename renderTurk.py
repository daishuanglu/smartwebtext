
import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score,precision_recall_curve, precision_score,recall_score
import matplotlib.pyplot as plt

from typing import Dict, Any
from fastDamerauLevenshtein import damerauLevenshtein
import pandas as pd
from tabulate import tabulate


f=open("MTurk/MTurkTemplate.txt",'r')
template=f.readlines()
f.close()

f=open("MTurk/MTurkHead.txt",'r')
head=f.readlines()
f.close()

f=open("MTurk/MTurkTail.txt",'r')
tail=f.readlines()
f.close()

df_batch_tic = pd.read_csv('MTurk/MTurk_eval_tics.csv')
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

    fo=open("MTurk/renderedTurk.txt",'w')
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
    df_predict = pd.merge(df_pred, df_gt, how='left', left_index=True, right_index=True).dropna()
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



def score(predictions, groundtruths, score_col, w, n_samples, propred_file=None):
    df_predict, df_gt_local = join_pred_gt(predictions, groundtruths)
    if propred_file is not None:
        propred_df = pd.read_csv(propred_file)
        cols = list(propred_df.columns)
        eval_col_id = np.argmax([damerauLevenshtein(score_col, col, similarity=True) for col in cols])
        eval_col = cols[eval_col_id]
    thresholds = np.linspace(0, 1, num=101)
    df_prec_rec = defaultdict(list)
    for th in thresholds:
        if propred_file is not None:
            df_prec_rec['recall'].append((propred_df[eval_col]> th ).mean())
        else:
            rec = recall_score(df_gt_local[w], df_predict[score_col] > th)
            df_prec_rec['recall'].append(rec * len(df_predict) / n_samples)
            # df_prec_rec['recall'].append(rec)
        df_prec_rec['precision'].append(precision_score(df_gt_local[w], df_predict[score_col]>th))
        df_prec_rec['threshold'].append(th)
    df_prec_rec = pd.DataFrame.from_dict(df_prec_rec)
    df_prec_rec = df_prec_rec.sort_values(by=['precision', 'recall'], ascending=False)
    idx = df_prec_rec.groupby('precision')['recall'].idxmax()
    df_prec_rec = df_prec_rec.loc[idx].sort_values(by='recall')
    return df_prec_rec


if __name__=="__main__":
    """
    render_mturk_html(df)
    """
    kws = ['analytics', 'innovation', 'technology']
    prediction_files = [
        'evaluation/local_topics_eval_predictions.csv',
        'evaluation/global_topics_eval_predictions.csv',
        'evaluation/edit_eval_predictions.csv',
        'evaluation/tte_sent_small_eval_predictions.csv',
    ]
    proprediction_files = [None, None, None, None]
    #proprediction_files = [
    #    'evaluation/local_topics_acct_eval_predictions.csv',  # 61% MTurk precision
    #    'evaluation/global_topics_acct_eval_predictions.csv',  # 61% MTurk precision
    #    'evaluation/edit_acct_eval_predictions.csv',  # 65% MTurk precision
    #    'evaluation/acct_tte_sent_small_eval_predictions.csv'  # 59% MTurk precision
    #]
    prediction_results = {}
    for pred_file, propred_file in zip(prediction_files, proprediction_files):
        df = pd.read_csv(pred_file)
        n_preds = len(df)
        df = process_predictions_df(df, kws)
        k = list(df.keys())[0]
        df[k] = (df[k], propred_file)
        prediction_results.update(df)
        print('%s model predictions on %d companies.' % (pred_file, n_preds))

    #  If you need to start a new MTurk batch task to collect GT labels.
    #  render_mturk_html(df)
    batch_file="./MTurkBatch/Batch_4766827_batch_results.csv"
    df_vote = append_MTurk_batch_response(batch_file)
    df_maj_vote = majority_vote(df_vote)
    df_maj_vote= df_maj_vote.drop_duplicates('tic').set_index('tic')
    nsamples = len(df_maj_vote)
    print("MTurk voted on %d companies." % nsamples)
    df_maj_vote.to_csv("MTurk/MTurk_maj_vote.csv")

    for w in kws:
        fig, ax = plt.subplots(1, 1)
        i=0
        for method, (pred_df, propred_file) in prediction_results.items():
            score_col = ':'.join([method, w])
            print(' ----------------- Threshold selector for ', score_col, ' ------------------')
            df_prec_rec = score(pred_df, df_maj_vote, score_col, w, nsamples, propred_file)
            _print_df(df_prec_rec)
            ax.plot(df_prec_rec['recall'], df_prec_rec['precision'], '.--', c=colormaps[i])
            i+=1
        ax.legend(list(prediction_results.keys()), loc='best')
        ax.set_title('%s:ROC' % w)
        ax.set_ylabel('precision')
        ax.set_xlabel('recall')
        plt.show(block=True)
