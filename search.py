import pandas as pd
from embedding import LocalTopicAsEmbedding, GlobalTopicAsEmbedding
from utils.string_utils import *
from utils.ticker_utils import ticker_finder
from fastDamerauLevenshtein import damerauLevenshtein
import time
from collections import defaultdict


EMB_MODEL_PATH = "config/acct_topic_emb.yaml" # only do eval for this model.
#EMB_MODEL_PATH = "config/topic_emb.yaml"

VAL_DATA_PATH = "data_model/tte_validation_data_unnorm.csv"
#MODE = 'validation'
MODE = 'eval'

SRC_DF = pd.read_csv("data_model/ner_cleaned_title.csv")

def get_flattened_kw_edit_sim_dataset(keywords, comp=None):
    nwords=len(keywords)
    src_df = SRC_DF.copy() if comp is None else\
            SRC_DF[SRC_DF['Company'].isin(comp)]
    src_df = src_df.sort_values(by=['Company'])
    df_scores = {}
    start_time=time.time()
    for ii,(t,c) in enumerate(zip(src_df['Title'], src_df['Company'])):
        t=str(t).lower()
        s=[max([damerauLevenshtein(w,kw,similarity=True) \
                for w in t.split('-')]) for kw in keywords]
        for j in range(nwords):
            w= keywords[j]
            if (c,w) not in df_scores:
                df_scores[(c, w)] = s[j]
            else:
                df_scores[(c, w)] = max(df_scores[(c, w)], s[j])
        if (ii+1) % 10000 ==0:
            print(ii+1,len(src_df),int(time.time()-start_time),' secs.')
    output_df = defaultdict(list)
    for c,w in df_scores.keys():
        output_df['company'].append(c)
        output_df['phrase'].append(w)
        output_df['scores'].append(df_scores[(c, w)])
    return pd.DataFrame.from_dict(output_df)


def get_kw_edit_sim_dataset(keywords, comp=None):
    nwords=len(keywords)
    src_df_ = SRC_DF.copy() if comp is None else\
            SRC_DF[SRC_DF['Company'].isin(comp)]
    comp_list=src_df_['Company'].tolist()
    scores=np.zeros((len(keywords),len(comp_list)))
    haskw=np.zeros((len(comp_list),len(keywords)))
    start_time=time.time()
    for ii,(t,c) in enumerate(zip(
            src_df_['Title'], src_df_['Company'])):
        t=str(t).lower()
        ic=comp_list.index(c)
        haskw[ii,:]=np.logical_or(haskw[ii,:], [kw in t for kw in keywords])
        s=[max([damerauLevenshtein(w,kw,similarity=True) \
                for w in t.split('-')]) for kw in keywords]
        for j in range(nwords):
            if s[j]>scores[j,ic]:
                scores[j,ic]=s[j]
        if (ii+1) % 10000 ==0:
            print(ii+1,len(src_df_),int(time.time()-start_time),' secs.')
    df=pd.DataFrame(scores.transpose(),columns=['edit_sim:'+k for k in keywords])
    df1 = pd.DataFrame(haskw, columns=['haskey:' + k for k in keywords])
    df['company']=[getcompanyname(c) for c in comp_list]
    df=pd.concat([df,df1],axis=1)
    df=df.groupby(['company'], sort=False, as_index=False)\
        [['edit_sim:'+k for k in keywords]+ ['haskey:' + k for k in keywords]].max().reset_index(drop=True)
    return df


def merge_ais_dataset(df0):
    #df0 = get_dataset(kws)
    # df0 = get_dataset(['accident', 'accounting'])
    print(df0)
    print(df0.columns)
    df_sic = pd.read_csv("ais/sic.csv")
    df_sic = df_sic[df_sic['ni'].notna()]
    df_sic = df_sic.drop_duplicates(["tic"], keep="last")
    df_sic = df_sic[acct_info]
    df0 = df0.merge(df_sic, how='left', on='tic').reset_index(drop=True)
    print(df0)
    print(df0.columns)
    df0.to_csv("accountingInfo_dataset.csv", index=False)
    return df0


def learn_comb_weights(df,kw,ref_col="edit_sim",th=0.8):
    ind=df[ref_col+':' + kw]>th
    xx0 = df[kw+'_s0'].tolist()
    xx1= df[kw].tolist()
    xx=np.array([xx0,xx1]).transpose()
    # run a simple OLS regression for w
    x= xx[ind,:]
    #s=np.zeros((x.shape[1],x.shape[1]))
    #for i in range(x.shape[0]):
    #    s+= np.outer(x[i,:],x[i,:])
    w = np.matmul(  np.linalg.inv(np.matmul(x.T,x) ), np.matmul( x.T , np.ones((x.shape[0],1)) ) )
    reg = np.matmul(x,np.atleast_2d(w))
    err= np.sqrt((reg -1)**2)
    err0=np.abs( reg-1)
    print("weights:", w)
    print('mse: ',err.mean(),", std. err:",err.std(),", max abs err.:", err0.max(), "min abs err:", err0.min())
    comb_score= np.matmul(xx,w)
    return w,comb_score[:,0]


def get_comb_score(df,keywords,ref_col="edit_sim",th=0.8):
    data = np.zeros((len(df), len(keywords)))
    for i,kw in enumerate(keywords):
        _, score=learn_comb_weights(df, kw, ref_col=ref_col, th=th)
        data[:,i]=score
    df0= pd.DataFrame(data,columns=['comb:'+k for k in keywords])
    df0['company']= df['company']
    return df0


def load_model(model_obj, model_path):
    model = model_obj(model_path)
    model._load()
    fn = lambda x, c: model.predict(x, c)
    print('model loaded.')
    return fn, model.get_local_names(), model.get_tickers()


def flattened_predict(df_val, predict_fn):
    pred = defaultdict(list)
    for c in sorted(set(df_val['Company'])):
        kws_local = df_val['phrase'][df_val['Company'] == c].tolist()
        score_local, kws_local = predict_fn(kws_local, [c])
        for i, w in enumerate(kws_local):
            pred['company'].append(c)
            pred['phrase'].append(w)
            pred['scores'].append(score_local[0, i])
    return pd.DataFrame.from_dict(pred)


def main():
    #eval_company = pd.read_csv("MTurk/MTurk_eval_tics.csv")['company']
    model_name = os.path.basename(EMB_MODEL_PATH).split('.yaml')[0]
    MODE_NAME = '{:s}_val_predictions' if MODE == 'validation' else '{:s}_eval_predictions'
    MODE_NAME = MODE_NAME.format(model_name)
    kws = ['analytic', 'innovation', 'technology']
    if MODE == 'validation':
        df_val = pd.read_csv(VAL_DATA_PATH)
        kws = sorted(set(df_val['phrase']))
    predict_fn, comp_list, tickers = load_model(LocalTopicAsEmbedding, EMB_MODEL_PATH)
    if MODE == 'validation':
        df_local = flattened_predict(df_val, predict_fn)
    else:
        score_local, kws_local = predict_fn(kws, None)
        df_local = pd.DataFrame(score_local, columns=['local_topics_sim:'+k for k in kws_local])


    predict_fn, _, _ = load_model(GlobalTopicAsEmbedding, EMB_MODEL_PATH)
    if MODE == 'validation':
        df_global = flattened_predict(df_val, predict_fn)
    else:
        score_global, kws_global = predict_fn(kws, None)
        df_global = pd.DataFrame(score_global, columns=["global_topics_sim:"+k for k in kws_global])

    if MODE =='validation':
        df_local.to_csv('evaluation/local_topics_%s.csv' % MODE_NAME, index=False)
        df_global.to_csv('evaluation/global_topics_%s.csv' % MODE_NAME, index=False)
        df_edit = get_flattened_kw_edit_sim_dataset(kws, comp=sorted(set(df_val['Company'])) )
        df_edit.to_csv('evaluation/edit_%s.csv' % MODE_NAME, index=False)
        return
    else:
        df_edit = get_kw_edit_sim_dataset(kws)

    df = pd.concat([df_local, df_global], axis=1)
    df['company'] = comp_list
    #df['link'] = find_links(comp_list, 'data_model/default_search_full.txt')
    df['tic'] = tickers
    df = df.merge(
        df_edit[['company']+[col for col in df.columns if 'haskey' in col]],
        how='left',
        on='company').reset_index(drop=True)

    local_topics_cols = [col for col in df.columns if 'local_topics' in col]
    global_topics_cols = [col for col in df.columns if 'global_topics' in col]
    common_cols = [col for col in df.columns if col not in local_topics_cols+global_topics_cols]
    df[local_topics_cols+ common_cols].to_csv('evaluation/local_topics_%s.csv' % MODE_NAME,index=False)
    df[global_topics_cols + common_cols].to_csv('evaluation/global_topics_%s.csv' % MODE_NAME, index=False)

    #df_edit['link'] = find_links(df_edit['company'].tolist(), 'data_model/default_search_full.txt')
    df_edit['tic'] = df_edit['company'].apply(lambda x: ticker_finder(x))
    df_edit.to_csv('evaluation/edit_%s.csv' % MODE_NAME,index=False)
    return

if __name__=='__main__':
    main()