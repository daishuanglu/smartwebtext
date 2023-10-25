import os
import pandas as pd
import numpy as np
from preprocessors.constants import *


UCF_RECG_SPLIT_PATH = \
    '{dataset_dir}/ucfTrainTestlist/{split}list{no:02d}.txt'
UCF_RECG_CLS_INDEX_PATH = \
    '{dataset_dir}/ucfTrainTestlist/classInd.txt'
UCF_RECG_SPLIT_FILE_NO = {'train': 3, 'test': 3}
UCF_RECG_SPLIT_HEADERS = {'train': ['vid_path', CLASS_ID_KEY], 'test': ['vid_path']}
UCF_RECG_TRAIN_SPLIT_CSV = 'data_model/ucf_recg_{split}.csv'
UCF_NUM_CLASSES = 101
INVALID_UCF_RECG_VIDS = ['PushUps/v_PushUps_g16_c04.avi',
                         'BlowingCandles/v_BlowingCandles_g05_c03.avi',
                         'SkyDiving/v_SkyDiving_g02_c01.avi',
                         'HorseRiding/v_HorseRiding_g14_c02.avi']


def ucf_recognition_splits_df(dataset_dir,
                              train_val_ratio=[0.95, 0.05],
                              **kwargs):
    class_index_fpath = UCF_RECG_CLS_INDEX_PATH.format(dataset_dir=dataset_dir)
    cls_ind = pd.read_csv(
        class_index_fpath, header=None, delimiter=' ', names=[CLASS_ID_KEY, CLASS_NAME])
    cls_ind.set_index(CLASS_NAME, inplace=True)
    dfs = []
    for split in ['train', 'test']:
        for i in range(3):
            split_fpath = UCF_RECG_SPLIT_PATH.format(dataset_dir=dataset_dir, split=split, no=i+1)
            print(split_fpath)
            df_split = pd.read_csv(
                split_fpath, header=None, delimiter=' ', names=UCF_RECG_SPLIT_HEADERS[split])
            df_split[CLASS_NAME] = df_split['vid_path'].apply(lambda x: os.path.dirname(x))
            if split == 'test':
                df_split[CLASS_ID_KEY] = df_split[CLASS_NAME].apply(
                    lambda x: cls_ind.loc[x][CLASS_ID_KEY])
            df_split = df_split[~df_split['vid_path'].isin(INVALID_UCF_RECG_VIDS)]
            df_split[CLIP_PATH_KEY] = df_split['vid_path'].apply(lambda x: os.path.join(dataset_dir, x))
            df_split[SAMPLE_ID_KEY] = df_split['vid_path'].apply(lambda x: os.path.basename(x).split('.')[0])
            df_split[SPLIT_KEY] = split
            dfs.append(df_split)
    dfs = pd.concat(dfs)
    df_test = dfs[dfs[SPLIT_KEY] == 'test']
    df_train = dfs[dfs[SPLIT_KEY] == 'train']
    df_train = df_train.drop_duplicates(subset=[SAMPLE_ID_KEY], keep='first')
    df_train[SPLIT_KEY] = df_train[SPLIT_KEY].apply(
        lambda x: np.random.choice(['train', 'val'], p=train_val_ratio))
    dfs = pd.concat([df_train, df_test])
    dfs[CLASS_ID_KEY] -= 1
    return dfs


def ucf_recognition(dataset_dir,
                    train_val_ratio=[0.95, 0.05],
                    **kwargs):
    dfs = ucf_recognition_splits_df(dataset_dir, train_val_ratio)
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df.to_csv(UCF_RECG_TRAIN_SPLIT_CSV.format(split=split), index=False)


VIDTXT_UCF_TRAIN_SPLIT_CSV = 'data_model/video_text_ucf_recg_{split}.csv'
VIDTXT_UCF_ALL_TEXTS = 'data_model/video_text_ucf_all_texts.txt'


def ucf_video_text(dataset_dir,
                   train_val_ratio=[0.95, 0.05],
                   **kwargs):
    dfs = ucf_recognition_splits_df(dataset_dir, train_val_ratio)
    all_texts = set()
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df[TARGET_KEY] = 1.0
        df[TEXT_KEY] = df[CLASS_NAME]
        df.to_csv(VIDTXT_UCF_TRAIN_SPLIT_CSV.format(split=split), index=False)
        all_texts.update(set(df[TEXT_KEY].dropna()))
    with open(VIDTXT_UCF_ALL_TEXTS, 'w') as f:
        for t in sorted(all_texts):
            f.write(t + '\n')