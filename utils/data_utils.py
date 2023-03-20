import os
import torch
import linecache
import pandas as pd
from scipy import sparse
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Callable
def randsample_df_by_group_rate(df, group_cols, rate):
    sample_info = (df.groupby(group_cols).size()*rate).round(0).astype(int)
    mapper = sample_info.to_dict()
    df = df.groupby(group_cols).apply(
        lambda x: x.sample(n=mapper.get(x.name),replace=False),
        ).reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def randsample_df_by_group_size(df, group_cols, size=10):
    sample_sizes = df.groupby(group_cols).size()
    sample_sizes = sample_sizes.apply(lambda x: size)
    sample_info = sample_sizes.round(0).astype(int)
    mapper = sample_info.to_dict()
    df = df.groupby(group_cols).apply(
        lambda x: x.sample(n=mapper.get(x.name),replace=False),
        ).reset_index(drop = True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, path, features_desc, nb_samples, sep=',',
                 clrCache=False, to_input_example=False, max_line = 10**7,
                 col_fns: Dict[str, Callable] = dict()):
        self.file = path
        self.len = nb_samples
        self.sep = sep
        self.clrCache = clrCache
        self.cnt = 0
        self.max_line = max_line
        self.features= features_desc
        if to_input_example:
            self.get_item_fn = lambda line: self.get_sbert_inp_feature(line)
        else:
            self.get_item_fn = lambda line: self.get_dict_feature(line)
        with open(self.file,'r') as f:
            self.headers = next(f).rstrip().split(sep)
        self.col_fns = col_fns

    def get_dict_feature(self, line):
        feature = {k: self.features[k](v) for k, v in zip(
                self.headers,
                line.rstrip().split(self.sep)
            ) if k in self.features.keys()}
        for output_key, fn in self.col_fns.items():
            feature[output_key] = fn(feature)
        return feature

    def get_sbert_inp_feature(self, line):
        features = [self.features[k](v) for k, v in zip(self.headers,
            line.rstrip().split(self.sep)) if k in self.features.keys()]
        example = [[] for _ in range(3)]
        for v, i in features:
            example[i].append(v)
        return {'texts': ['[DOC]'.join(example[0]), example[1]], 'label': float(example[-1][0])}

    def __getitem__(self, index):
        irow = index + 2
        line = linecache.getline(self.file, irow)
        if self.cnt == self.max_line and self.clrCache:
            linecache.clearcache()
            self.cnt=0
        try:
            self.cnt += 1
            feature = self.get_item_fn(line)
        except Exception as e:
            print(e)
            print(line)
            print(index,  irow, self.file)
            assert False, 'error captured.'
        return feature

    def __len__(self):
        return self.len


def collate_dict(samples):
    result = {}
    for sample in samples:
        for key in sample:
            result.setdefault(key, []).append(sample[key])
    for key in result:
        if isinstance(result[key][0], float):
            result[key] = torch.FloatTensor(result[key])
    return result


def get_sbert_csv_evaluator_dataset(data_path, ref_col, context_col, query_col,
                                    score_col, special_sep="[DOC]", limit=None):
    df= pd.read_csv(data_path)
    df = df if limit is None else df[:limit]
    print("%s: %d samples." % (data_path, len(df)))
    sent1 = (df[ref_col]+special_sep+df[context_col]).to_list()
    sent2 = df[query_col].to_list()
    scores = df[score_col].to_list()
    return sent1, sent2, scores


def get_context_csv_data_loader(data_path, features_desc, batch_size=2048,
                                clear_cache=True, shuffle=False, sep=',', max_line=10**7, limit=None,
                                to_sbert_input=False, col_fns=dict()):
    nb_samples = sum(1 for _ in open(data_path, 'r'))-1 if limit is None else limit
    print("%s: %d samples." % (data_path, nb_samples))
    if limit is not None:
        df = pd.read_csv(data_path)[:limit]
        os.makedirs('debug_info', exist_ok=True)
        limit_data_path = os.path.join(
            'debug_info',
            os.path.basename(data_path).replace('.csv',"_limit.csv"))
        df.to_csv(limit_data_path, index=False)
        data_path = limit_data_path
    tensor_data = CSVDataset(
        path = data_path,
        features_desc=features_desc,
        nb_samples=nb_samples,
        clrCache=clear_cache,
        sep=sep,
        to_input_example=to_sbert_input,
        max_line=max_line,
        col_fns=col_fns)
    return torch.utils.data.DataLoader(
        tensor_data,
        collate_fn=collate_dict,
        batch_size=batch_size,
        shuffle=shuffle)


def get_data_loader(dataset, mask, features, name="_", batch_size=2048, temp_data_path="./data_model", max_line=10**7):
    data_path = temp_data_path+'_'+name
    if isinstance(dataset, sparse.csr_matrix):
        ind_item, ind_user = mask.nonzero()
        #values= dataset.data.tolist()
    elif isinstance(dataset, np.memmap):
        inds = np.argwhere(mask!=0)
        ind_item, ind_user = inds[:,0], inds[:,1]
    else:
        assert False, 'dataset format has to be a sparse csr matrix, or numpy memmap'
    ifile = 0
    flist = [data_path + '_' + str(ifile)]
    f = open(flist[-1], 'w')
    f.write('user_id,item_id,score\n')
    for il, (u, i) in enumerate(zip(ind_user, ind_item)):
        v = dataset[i, u]
        f.write(','.join([str(u), str(i), str(v)]) + '\n')
        if il % max_line == 0 and il>0:
            print( il // max_line, '0M lines.')
            ifile += 1
            flist.append(data_path + '_' + str(ifile))
            f.close()
            f = open(flist[-1], 'w')
            f.write('user_id,item_id,score\n')
    f.close()
    nb_samples = len(ind_user)
    tensor_data = CSVDataset(flist, nb_samples=nb_samples, clrCache=len(flist)>1)
    dataloader= torch.utils.data.DataLoader(
        tensor_data,
        collate_fn=collate_dict,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader