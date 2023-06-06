import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

def flatten_cols(df_pred, df_val):
    df_val_ = df_val.copy().set_index(['Company','phrase'])
    method = list(set(col.split(':')[0] for col in df_pred.columns if (':' in col))-{'haskey'})[0]
    word_cols = [col for col in df_pred.columns if (':' in col) and ('haskey' not in col)]
    df = defaultdict(list)
    companies = df_pred['company']
    df_pred.set_index('company', inplace=True)
    for w in word_cols:
        word = w.split(':')[1]
        for c in companies:
            if (c, word) not in df_val_.index:
                continue
            df['company'].append(c)
            df['scores'].append(df_pred.loc[c][w])
            df['phrase'].append(word)
    return pd.DataFrame.from_dict(df), method


def labeling(df_pred, df_val, query_col='phrase'):
    query_col = query_col.lower()
    r = {col: col.lower() for col in df_pred.columns}
    df_pred = df_pred.rename(r, axis=1)
    df_val = df_val.copy()
    r = {col: col.lower() for col in df_val.columns}
    df_val = df_val.rename(r, axis=1)
    df_val = df_val.set_index(['company', query_col])
    df_pred = df_pred.set_index(['company', query_col])
    common_row_ids = df_val.index.intersection(df_pred.index)
    n_total = len(common_row_ids)
    coverage = n_total / len(df_val)
    df_val = df_val.loc[common_row_ids]
    df_pred = df_pred.loc[common_row_ids]
    df_val = df_val.rename({'scores': 'label'}, axis=1)
    df_pred = df_pred.join(df_val,on=['company', query_col])
    df_stats = df_pred.reset_index()
    print('Generated {:d} words, {:d} companies, {:d} common evaluation samples, coverage {:.2f}.'.format(
        len(set(df_stats[query_col])), len(set(df_stats['company'])) , len(df_stats), coverage ))
    return df_pred, coverage


def _plot(ax, pred, gt, c, l, coverage):
    #pred = np.maximum(pred, 0.0)
    #pred = np.minimum(pred, 1.0)
    precision, recall, thresholds = precision_recall_curve(gt.tolist(), pred.tolist(), pos_label=1)
    recall = coverage * recall
    map = average_precision_score(
        gt, pred, average='micro', pos_label=1)
    ax.plot(recall, precision, color=c, label=l)
    leg = l + '- mAP:' + str(map)
    return ax, leg

def load_val_data(path):
    df_val = pd.read_csv(path)
    df_val['scores'] = df_val['scores'].apply(lambda x: (0 if x == 0 else 1))
    return df_val

def isflattened(df):
    return not any([':' in col for col in df.columns])

if __name__ == "__main__":
    """
    Because topic modelings preprocssing used the keyphrase extraction to improve quality of word frequencies.
    If not use keyphrase extraction, we will introduce Large-scale (word-to-num_of_topic*num_of_company) embedding matrix (>10GB) which is impossible to learn.
    With kp extraction Some non-related or invalid words are removed. that's why topic models tends to 
    have low recall as it is not considering all the non-kp words as an embedding. 
    While topic models maintains a high precision, even though we restricted the min_word_cnt =1, 
    the model still discarded non-kp words but may be meaningful. Comparably a seq2seq model is more generic 
    and scalable for this type of problem.
    """

    colormaps = ['orange', 'purple', 'blue', 'red', 'green', 'yellow', 'black', 'brown']
    prediction_files = [
        'evaluation/local_topics_val_predictions.csv',
        'evaluation/global_topics_val_predictions.csv',
        #'evaluation/edit_val_predictions.csv',
        'evaluation/tte_val_predictions.csv',
        'evaluation/tte_sent_small_val_predictions.csv',
        #'evaluation/acct_tte_sent_small_val_predictions.csv'
    ]
    validation_data_files = [
        "data_model/tte_validation_data_unnorm.csv",
        "data_model/tte_validation_data_unnorm.csv",
        #"data_model/tte_validation_data_unnorm.csv",
        "data_model/tte_validation_data_unnorm.csv",
        "data_model/tte_validation_data.csv",
        #"data_model/acct_tte_validation_data.csv",
    ]
    query_cols = [
        'phrase',
        'phrase',
        'phrase',
        #'phrase',
        'Text',
        #'Text'
    ]
    fig, ax = plt.subplots(1, 1)
    legend = []
    for i, (predictions, val_data_path, query_col) in enumerate(
            zip(prediction_files, validation_data_files, query_cols)):
        df_pred = pd.read_csv(predictions)
        df_val = load_val_data(val_data_path)
        if not isflattened(df_pred):
            df_pred, method = flatten_cols(df_pred, df_val)
        else:
            method = os.path.basename(predictions).split('_val_predictions')[0]
        print('+ ', method)
        df_pred, coverage = labeling(df_pred, df_val, query_col)
        ax, leg = _plot(ax, df_pred['scores'], df_pred['label'].astype(int), colormaps[i], method, coverage=1.0)
        legend.append(leg)
    ax.legend(legend, loc='best')
    ax.set_title('validation set ROC'.format())
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')

    plt.show(block=True)

    #df['company'] = sorted(set(dataset['Company']))
    #df['tic'] = [ticker_finder(c) for c in df['company']]
    #phrases = sorted(set(dataset['phrase'])-{'company'})
    #for w in phrases:
    #    df[w] = [-1 for _ in range(len(df['company']))]
    #print('%d companies, %d phrases.' % (len(df['company']), len(phrases)))
    #for ir, row in dataset.iterrows():
    #    if row['phrase'] in phrases:
    #        df[row['phrase']][df['company'].index(row['Company'])] = row['scores'] if row['scores']==0 else 1
    #pd.DataFrame.from_dict(df).to_csv("evaluation/validation_groundtruths.csv", index=False)