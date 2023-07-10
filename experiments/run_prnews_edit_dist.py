import os
import pandas as pd
import numpy as np
import time
from fastDamerauLevenshtein import damerauLevenshtein
from utils import ticker_utils
from preprocessors import pipelines

COMBINED_TEXT_COL = 'Text'
TEST_DATA_PATH = "data_model/prnews_tte_collab_filter_validation.csv"
MODE_NAME = 'val'


def get_kw_edit_sim_dataset(src_df, keywords, comp=None):
    keywords = [kw.lower() for kw in keywords]
    src_df_ = src_df.copy() if comp is None else\
            src_df[src_df['Company'].isin(comp)]
    comp_list=src_df_['Company'].unique().tolist()
    print('%d companies.' % len(comp_list))
    scores=np.zeros((len(keywords),len(comp_list)))
    sentences = [['' for _ in range(len(comp_list))] for _ in range(len(keywords))]
    haskw=np.zeros((len(comp_list),len(keywords)))
    start_time=time.time()
    for ii,(context,company) in enumerate(zip(src_df_['Title']+' '+src_df_['Text'], src_df_['Company'])):
        context = str(context).lower()
        ic=comp_list.index(company)
        haskw[ic,:]=np.logical_or(haskw[ic,:], [kw in context for kw in keywords])
        s = [max([damerauLevenshtein(
            w,kw,similarity=True) for w in context.split()]) for kw in keywords]
        for j in range(len(keywords)):
            if s[j]>scores[j,ic]:
                scores[j,ic]=s[j]
                sentences[j][ic] = context
        if (ii+1) % 10000 ==0:
            print(ii+1,len(src_df_),int(time.time()-start_time),' secs.')
    df_sent = pd.DataFrame(sentences, index=['edit_sim_sent:'+w for w in keywords]).T
    df = pd.DataFrame(scores.transpose(),columns=['edit_sim:'+k for k in keywords])
    df = pd.concat([df, df_sent], axis=1)
    df1 = pd.DataFrame(haskw, columns=['haskey:' + k for k in keywords])
    #df['company']=[string_utils.getcompanyname(c) for c in comp_list]
    df['company'] = comp_list
    df=pd.concat([df,df1],axis=1)
    return df


if __name__ == '__main__':
    os.makedirs(pipelines.PRNEWS_EVAL_DIR, exist_ok=True)
    src_df = pd.read_csv(
        TEST_DATA_PATH, sep=pipelines.PRNEWS_DATA_SEP, dtype=str, parse_dates=False, na_values=[], keep_default_na=False)
    test_keywords = ['analytics', 'innovation', 'technology']
    df_predictions = get_kw_edit_sim_dataset(src_df, test_keywords, comp=None)
    df_predictions['tic'] = df_predictions['company'].apply(lambda x: ticker_utils.ticker_finder(x))
    df_predictions.to_csv(os.path.join(
        pipelines.PRNEWS_EVAL_DIR, 'edit_%s_predictions.csv' % MODE_NAME), index=False)
