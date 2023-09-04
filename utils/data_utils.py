import os
import torch
import linecache
import pandas as pd
from scipy import sparse
import numpy as np
from typing import Dict, Callable


class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, path, features_desc, nb_samples, sep=',', flat_cols=False,
                 clrCache=False, max_line = 10**7, col_fns: Dict[str, Callable] = dict()):
        self.file = path
        self.len = nb_samples
        self.sep = sep
        self.clrCache = clrCache
        self.cnt = 0
        self.max_line = max_line
        self.features= features_desc
        self.flat_dict = flat_cols
        self.get_item_fn = lambda line: self.get_dict_feature(line)
        with open(self.file, 'r') as f:
            self.headers = next(f).rstrip().split(sep)
        self.col_fns = col_fns

    def get_dict_feature(self, line):
        feature = {k: self.features[k](v) for k, v in zip(
                self.headers,
                line.split(self.sep)
            ) if k in self.features.keys()}
        #print('before', feature['ocr_word_confidence'][0])
        for output_key, fn in self.col_fns.items():
            col_value = fn(feature)
            if self.flat_dict and isinstance(col_value, dict):
                feature.update(col_value)
            else:
                feature[output_key] = col_value
        #print('after ', feature['ocr_word_confidence'][0])
        return feature

    def __getitem__(self, index):
        irow = index + 2
        line = linecache.getline(self.file, irow)
        line = line.strip('"\n')
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
            assert False, e.__repr__()
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


def get_context_csv_data_loader(data_path, features_desc, batch_size=2048,
                                clear_cache=True, shuffle=False, sep=',', max_line=10**7, limit=None,
                                col_fns=dict(), flat_cols=False):
    nb_samples = sum(1 for _ in open(data_path, 'r'))-1
    nb_samples = min(nb_samples, limit) if limit is not None else nb_samples
    print("%s: %d samples." % (data_path, nb_samples))
    if limit is not None:
        df = pd.read_csv(
            data_path, sep=sep, dtype=str, keep_default_na=False, na_values=[], parse_dates=False)
        df = df[:limit] if limit < len(df) else df
        print("%s: %d samples." % (data_path, nb_samples))
        os.makedirs('ig_debug_info', exist_ok=True)
        limit_data_path = os.path.join(
            'ig_debug_info',
            os.path.basename(data_path).replace('.csv',"_limit.csv"))
        df.to_csv(limit_data_path, sep=sep, index=False)
        data_path = limit_data_path
    tensor_data = CSVDataset(
        path = data_path,
        features_desc=features_desc,
        nb_samples=nb_samples,
        clrCache=clear_cache,
        sep=sep,
        max_line=max_line,
        col_fns=col_fns,
        flat_cols=flat_cols)
    return torch.utils.data.DataLoader(
        tensor_data,
        collate_fn=collate_dict,
        batch_size=batch_size,
        shuffle=shuffle)
