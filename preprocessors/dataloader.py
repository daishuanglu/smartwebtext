import linecache
import glob
import itertools
import operator
import bisect
import pandas as pd
import time
from typing import Tuple, List, Dict
from tqdm import tqdm
import numpy as np
import json


def na_pandas_rows(df, key_cols):
    if not key_cols:
        return pd.Series(False, index=df.index)
    return df.mask(df == '')[key_cols].isna().any(axis=1)


class BaseTextFile():
    def __init__(self,
                 fpath,
                 sep,
                 types: Dict = dict(),
                 col_fns: List[Tuple] = [],
                 must_contain: List[str] = []):
        self.sep = sep
        self.file_paths = sorted(glob.glob(fpath))
        self.headers = linecache.getline(self.file_paths[0], 1).strip().split(sep)
        self.nfiles = len(self.file_paths)
        self.nrows = [len(open(
            fname, 'r', encoding='utf-8').readlines())-1 for fname in self.file_paths]
        self.nrows_cumsum = [0] + list(itertools.accumulate(self.nrows, operator.add))
        missing_types = set(types.keys()) - set(self.headers)
        assert len(missing_types) == 0,\
            'Cannot load types for column %s. \n Text file headers=%s \n Loading column types=%s' % \
            (','.join(missing_types), str(self.headers), str(list(types.keys())))
        self.types = {col: fn for col, fn in types.items() if col in self.headers}
        self.types = {**self.types, **{h: str for h in set(self.headers) - set(types.keys())}}
        self.col_fns = col_fns
        self.must_contain_cols = must_contain

    def line(self, id):
        ifile = bisect.bisect(self.nrows_cumsum, id) - 1
        line_no = id - self.nrows_cumsum[ifile] + 1
        line = linecache.getline(self.file_paths[ifile], line_no + 1)
        #print(id, line_no, ifile)
        return line

    def row(self, id):
        # type itself is an element-wise preprocessing function.
        # col fns is a list of tuple(function_name, function) applied to a row dict.
        #if id == 2376 or id == 2389:
        #    print()
        line = self.line(id)
        s = {h: self.types[h](s) for h, s in zip(
            self.headers, line.strip().split(self.sep))}
        if len(s) != len(self.headers):
            print('Skipped line ID=%d. Number of line splits does not match the '\
                  'number of headers. line=%s' % (id, line.__repr__()))
            return False
        if not any([v for v in s.values()]):
            print('Skipped line ID=%d, Empty line values.' % id)
            return False
        for col_name, fn in self.col_fns:
            s[col_name] = fn(s)
        if not all([s[key].strip() for key in self.must_contain_cols]):
            print('Skipped line ID=%d. Missing must-contain key values. line=%s', (id, str(s)))
            return False
        return s

    def vocab(self, cols, vocab_path):
        vocab = {}
        i = 0
        for id in tqdm(
                range(sum(self.nrows)),
                desc='create vocabulary based on columns %s' % ','.join(cols)):
            s = self.row(id)
            if s:
                for col in cols:
                    if s[col].lower() not in vocab.keys():
                        vocab[s[col].lower()] = i
                        i+=1
        with open(vocab_path, mode='w', encoding='utf-8') as f:
            json.dump(vocab, f)

    def write(self, output_files, split_ratio, cols, explode_sep=None, sep='\t'):
        assert sep != ' ', 'Separator must not be empty char.'
        assert sep != '\n', 'Separator must not be line breaks.'
        fps = [open(
            output_file, mode='w', encoding='utf-8') for output_file in output_files]
        for fp in fps:
            fp.write(sep.join(cols)+'\n')
        removed_rows = 0
        for id in tqdm(range(sum(self.nrows)), desc='write splits'):
            s = self.row(id)
            if not s:
                removed_rows +=1
                continue
            texts = [[sp for sp in s[col].split(explode_sep) if sp.strip()
                      ] if explode_sep else [s[col]] for col in cols]
            for line_list in itertools.product(*texts):
                fps[np.random.choice(range(len(fps)), 1, replace=True, p=split_ratio)[0]].write(
                    sep.join(line_list)+'\n')
        print('Skipped %d / %d rows.' % (removed_rows, sum(self.nrows)))


class PandasTextFile(BaseTextFile):
    def __init__(self, fpath, sep, types=None, col_fns: List[Tuple] = [], must_contain=[]):
        super().__init__(fpath,sep, types, col_fns, must_contain)

    def _apply_col_fns(self, df):
        # type itself is an element-wise preprocessing function.
        # col fns is a list of tuple(function_name, function) applied to a row dict.
        start_time = time.time()
        for c, type in self.types.items():
            print('Single column value type function for %s' % c)
            df[c] = df[c].map(type, na_action='ignore')
            print('%d secs.' % int(time.time()-start_time))
        for col_name, fn in self.col_fns:
            print('Cross row function %s' % col_name)
            df[col_name] = df.apply(fn, axis=1)
            print('%d secs.' % int(time.time() - start_time))
        return df

    def file(self, i):
        df = pd.read_csv(
            self.file_paths[i % self.nfiles],
            dtype=str,
            keep_default_na=False, na_values=[], parse_dates=False)
        df = self._apply_col_fns(df)
        return df[~na_pandas_rows(df, self.must_contain_cols)]

    def all(self):
        dfs = [pd.read_csv(
            self.file_paths[i], dtype=str, parse_dates=False) for i in range(self.nfiles)]
        dfs = pd.concat(dfs, ignore_index=True)
        dfs = self._apply_col_fns(dfs)
        return dfs[~na_pandas_rows(dfs, self.must_contain_cols)]
