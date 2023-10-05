
from preprocessors.constants import *
from preprocessors.a2d import *
from preprocessors.bsds import *
from preprocessors.kth import *
from preprocessors.prnews import *
from preprocessors.ucf import *
from preprocessors.hmdb51 import *
from preprocessors.something_something_v2 import *


VID_TXT_COLS = [SAMPLE_ID_KEY, CLIP_PATH_KEY, SPLIT_KEY, CLASS_NAME, FRAME_ID_KEY]
VID_RECG_COLS = [SAMPLE_ID_KEY,
                 CLIP_PATH_KEY,
                 SPLIT_KEY,
                 CLASS_NAME,
                 CLASS_ID_KEY,
                 FRAME_ID_KEY]
VID_RECG_TRAIN_SPLIT_CSV = 'data_model/video_recognition_{split}.csv'


def video_recognition(dataset_configs):
    dfs = []
    for ds_name, config in dataset_configs.items():
        fn = globals().get(config['pipeline_fn'])
        df = fn(**config)
        cols = df.columns.intersection(VID_RECG_COLS)
        df = df[cols]
        df[DATASET_KEY] = ds_name
        for col in set(VID_RECG_COLS) - set(cols):
            df[col] = ''
        dfs.append(df)
    dfs = pd.concat(dfs)
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df.to_csv(VID_RECG_TRAIN_SPLIT_CSV.format(split=split), index=False)


VIDTXT_TRAIN_SPLIT_CSV = 'data_model/video_text_mixed_{split}.csv'
VIDTXT_ALL_TEXTS = 'data_model/video_text_mixed_all_texts.txt'


def mixed_video_text(dataset_configs):
    dfs = []
    for ds_name, config in dataset_configs.items():
        fn = globals().get(config['pipeline_fn'])
        df = fn(**config)
        cols = df.columns.intersection(VID_TXT_COLS)
        df = df[cols]
        df[DATASET_KEY] = ds_name
        for col in set(VID_TXT_COLS) - set(cols):
            df[col] = ''
        dfs.append(df)
    dfs = pd.concat(dfs)
    all_texts = set()
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df[TEXT_KEY] = df[CLASS_NAME]
        df.to_csv(VIDTXT_TRAIN_SPLIT_CSV.format(split=split), index=False)
        all_texts.update(set(df[TEXT_KEY].dropna()))
    with open(VIDTXT_ALL_TEXTS, 'w') as f:
        for t in sorted(all_texts):
            f.write(t + '\n')

