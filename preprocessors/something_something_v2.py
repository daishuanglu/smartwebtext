import os
import json
import pandas as pd

from preprocessors.constants import *


SS_V2_CSV = 'data_model/kth_actions.csv'
SS_V2_SPLIT_CSV = 'data_model/something-something-v2_{split}.csv'
SS_V2_SPLITS = ['train', 'validation', 'test']
SS_V2_CLIP_PATH = '{root}/{id}.mp4'
SS_V2_RECG_TRAIN_SPLIT_CSV = 'data_model/ss_v2_recg_{split}.csv'
SS_V2_VIDTXT_TRAIN_SPLIT_CSV = 'data_model/video_text_ss_v2_vidtxt_{split}.csv'
SS_V2_VIDTXT_ALL_TEXTS = 'data_model/video_text_ss_v2_all_texts.txt'


def something_something_v2_splits_df(dataset_dir, **kwargs):
    label_dir = os.path.join(dataset_dir, 'labels')
    with open(os.path.join(label_dir, 'labels.json'), 'r') as f:
        label_map = json.load(f)
    split_fpath = os.path.join(label_dir, '{split}.json')
    dfs = []
    for split in SS_V2_SPLITS:
        with open(split_fpath.format(split=split), 'r') as f:
            samples = json.load(f)
            df = pd.DataFrame(samples)
            df[SPLIT_KEY] = split.replace('validation', 'val')
            df[TEXT_KEY] = df.apply(
                lambda x: x['template'].replace("[something]", "{}").format(*x['placeholders']),
                axis=1
            )
            df[CLASS_NAME] = df['template'].apply(lambda x: x.replace('[something]', 'something'))
            df[SAMPLE_ID_KEY] = df['id']
            df[CLASS_ID_KEY] = df[CLASS_NAME].map(label_map)
            df[CLIP_PATH_KEY] = df[id].apply(
                lambda x: SS_V2_CLIP_PATH.format(root=dataset_dir, id=x))
        dfs.append(df)
    return pd.concat(dfs)


def ss_v2_recognition(dataset_dir):
    dfs = something_something_v2_splits_df(dataset_dir)
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df.to_csv(SS_V2_RECG_TRAIN_SPLIT_CSV.format(split=split), index=False)


def ss_v2_video_text(dataset_dir):
    dfs = something_something_v2_splits_df(dataset_dir)
    all_texts = set()
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df[TARGET_KEY] = 1.0
        df[TEXT_KEY] = df[CLASS_NAME]
        df.to_csv(SS_V2_VIDTXT_TRAIN_SPLIT_CSV.format(split=split), index=False)
        all_texts.update(set(df[TEXT_KEY].dropna()))
    with open(SS_V2_VIDTXT_ALL_TEXTS, 'w') as f:
        for t in sorted(all_texts):
            f.write(t + '\n')