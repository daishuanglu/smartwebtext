import pandas as pd
import lightgbm as lgb
from collections import Counter
import random
import numpy as np
import torch

label_num = {'Slight Injury': 0, 'Fatal injury': 2, 'Serious Injury': 1}
label_res = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}


def btraining(df_train, targets, label_name):
    split = pd.Series(index=targets.index).apply(lambda x: 'train' if random.random() > 0.2 else 'val')
    targets = targets == label_name
    train_data = lgb.Dataset(df_train[split == 'train'], label=targets[split == 'train'])
    val_data = lgb.Dataset(df_train[split == 'val'], label=targets[split == 'val'])
    num_round = 100
    param = {
        'num_leaves': 31, 
        'objective': 'binary',
        #'objective': 'multiclass', 
        #'num_classes': 3, 
        'metric': ['auc', 'binary_logloss'],
        'verbose': 1}
    bst = lgb.train(param, train_data, num_round, valid_sets=[val_data])
    #bst.save_model('model.txt')
    return bst


def training(df_train, targets):
    split = pd.Series(index=targets.index).apply(lambda x: 'train' if random.random() > 0.2 else 'val')
    train_data = lgb.Dataset(df_train[split == 'train'], label=targets[split == 'train'])
    val_data = lgb.Dataset(df_train[split == 'val'], label=targets[split == 'val'])
    num_round = 100
    param = {
        'num_leaves': 31, 
        #'objective': 'binary',
        'objective': 'multiclass', 
        'num_classes': 3, 
        #'metric': ['auc', 'binary_logloss'],
        'verbose': 1}
    bst = lgb.train(param, train_data, num_round, valid_sets=[val_data])
    return bst


if __name__ == '__main__':
    df_train = pd.read_csv('data/rta/train.csv', na_values=['na', 'NA', 'N/A'])
    df_train = df_train.set_index('Num')
    print(df_train.columns)
    print(df_train.iloc[0])
    targets = df_train['Accident_severity']
    print(Counter(targets))
    df_train = df_train.drop(columns=['Accident_severity'])
    for col in df_train.columns:
        if col != 'Num':
            df_train[col] = df_train[col].apply(lambda x: hash(str(x)) % 10000)
    print('preprocessed data:')
    print(df_train.iloc[0])
    targets = targets.map(label_num)
    print(targets)
    split = pd.Series(index=targets.index).apply(lambda x: 'train' if random.random() > 0.2 else 'val')
    train_data = lgb.Dataset(df_train[split == 'train'], label=targets[split == 'train'])
    val_data = lgb.Dataset(df_train[split == 'val'], label=targets[split == 'val'])
    bst = training(df_train, targets)
    
    df_test = pd.read_csv('data/rta/test.csv')
    for col in df_test.columns:
        if col != 'Num':
            df_test[col] = df_test[col].apply(lambda x: hash(str(x)) % 10000)
    #test_data = lgb.Dataset(df_test)
    df_test = df_test.set_index('Num')
    print(df_test.sort_index())
    ypred = bst.predict(df_test, num_iteration=bst.best_iteration)
    print('final predictions:')
    print(ypred)
    ind = np.argmax(ypred, axis=1)
    print(ind, Counter(ind))
    binpred = torch.nn.functional.one_hot(torch.tensor(ind).long(), num_classes=3).numpy()
    output = pd.DataFrame(data=binpred, index=df_test.index, columns=['Slight', 'Serious', 'Fatal'])
    print(len(output))
    output = output[~output.index.duplicated(keep='first')]
    print(len(output))
    output.astype(int).to_csv('rta_test_output.csv')
    
    