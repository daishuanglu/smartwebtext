"""
There are totally 153 files in this folder,
[action]_test_split[1-3].txt  corresponding to three splits reported in the paper.
The format of each file is
[video_name] [id]
The video is included in the training set if id is 1
The video is included in the testing set if id is 2
The video is not included for training/testing if id is 0
There should be 70 videos with id 1 , 30 videos with id 2 in each txt file.
"""
import os
import pandas as pd
import glob

from preprocessors.constants import *


HMDB51_CLIP_PATH = '{root}/{action}/{vid}'
HMDB51_ANNO_DIR = '{root}/testTrainMulti_7030_splits'
HMDB51_ANNO_FILE = '{action}_test_split{no:01d}.txt'
HMDB51_SPLIT_MAP = {'1': 'train', '0': 'val', '2': 'test'}
HMDB51_RECG_TRAIN_SPLIT_CSV = 'data_model/recg_hmdb51_{split}.csv'
HMDB51_VIDTXT_TRAIN_SPLIT_CSV = 'data_model/vidtxt_hmdb51_{split}.csv'
HMDB51_VIDTXT_ALL_TEXTS = 'data_model/vidtxt_hmdb51_all_texts.txt'
INVALID_HMDB51_RECG_VIDS = []


def hmdb51_splits_df(dataset_dir, **kwargs):
    anno_dir = HMDB51_ANNO_DIR.format(root=dataset_dir)
    files = sorted(glob.glob(os.path.join(anno_dir, '*.txt')))
    actions = []
    dfs = []
    for f in files:
        a = os.path.basename(f).split('_test_')[0]
        if a not in actions:
            actions.append(a)
        for i in range(1, 4):
            fsplit = os.path.join(anno_dir, HMDB51_ANNO_FILE.format(action=a, no=i))
            df_split = pd.read_csv(fsplit, sep=' ', dtype=str, header=None, names=['vid', 'split_no', 'empty'])
            df_split[SPLIT_KEY] = df_split['split_no'].map(HMDB51_SPLIT_MAP)
            df_split[CLASS_NAME] = a
            df_split = df_split[~df_split['vid'].isin(INVALID_HMDB51_RECG_VIDS)]
            df_split[CLASS_ID_KEY] = actions.index(a)
            df_split[SAMPLE_ID_KEY] = df_split['vid'].apply(lambda x: x.split('.')[0])
            df_split[CLIP_PATH_KEY] = df_split['vid'].apply(
                lambda x: HMDB51_CLIP_PATH.format(root=dataset_dir, action=a, vid=x))
            df_split[TEXT_KEY] = df_split['vid'].apply(lambda x: x.split('_np')[0])
            dfs.append(df_split)
    dfs = pd.concat(dfs)
    dfs = dfs.drop_duplicates(subset=[SPLIT_KEY, SAMPLE_ID_KEY], keep='first')
    return dfs


def hmdb51_recognition(dataset_dir,**kwargs):
    dfs = hmdb51_splits_df(dataset_dir, **kwargs)
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df.to_csv(HMDB51_RECG_TRAIN_SPLIT_CSV.format(split=split), index=False)


def hmdb51_video_text(dataset_dir, **kwargs):
    dfs = hmdb51_splits_df(dataset_dir,  **kwargs)
    all_texts = set()
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df[TARGET_KEY] = 1.0
        df[TEXT_KEY] = df[CLASS_NAME]
        df.to_csv(HMDB51_VIDTXT_TRAIN_SPLIT_CSV.format(split=split), index=False)
        all_texts.update(set(df[TEXT_KEY].dropna()))
    with open(HMDB51_VIDTXT_ALL_TEXTS, 'w') as f:
        for t in sorted(all_texts):
            f.write(t + '\n')


if __name__ == '__main__':
    dataset_dir = 'D:/video_datasets/hmdb51'
    df = hmdb51_splits_df(dataset_dir)
    print(df)