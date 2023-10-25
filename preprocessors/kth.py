import os
import collections
import pandas as pd
from preprocessors.constants import *


KTH_ACTION_SRC_CSV = 'data_model/kth_actions.csv'
KTH_ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
KTH_ACTION_DATA_CSV_SEP = ','
KTH_SPLIT_CSV = 'data_model/kth_actions_{split}.csv'
KTH_FRAME_FEATURE_SEP = ';'
#KTH_ACTION_HOME = '/home/shuangludai/KTHactions'
KTH_VIDEO_FILE = '{action}/person{pid}_{action}_{var}_uncomp.avi'


def kth_action_recg_splits_df(kth_action_home, clsname_map_path, **kwargs):
    fpath = os.path.join(kth_action_home, '00sequences.txt')
    splits = {}
    df = collections.defaultdict(list)
    with open(fpath, 'r') as f:
        for line in f.readlines():
            if 'training:' in line.lower():
                splits.update({pid: 'train' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'validation:' in line.lower():
                splits.update({pid: 'val' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'test:' in line.lower():
                splits.update({pid: 'eval' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'frames' in line.lower():
                vid_file, kfs = line.strip().split('frames')
                vid_file = vid_file.strip()
                fid_ranges = [frange.split('-') for frange in kfs.strip().split(', ')]
                fids = []
                for start, end in fid_ranges:
                    fids += list(range(int(start), int(end)))
                df[FRAME_ID_KEY].append(FRAME_ID_SEP.join(list(map(str, fids))))
                pid, action, var = vid_file.split('_')
                pid = pid.split('person')[1]
                #df[PERSON_ID_KEY].append(int(pid))
                df[CLASS_NAME].append(action)
                df[CLASS_ID_KEY].append(KTH_ACTIONS.index(action))
                vid_file_basepath = KTH_VIDEO_FILE.format(action=action, pid=pid, var=var)
                vid_file_path = os.path.join(kth_action_home, vid_file_basepath)
                df[CLIP_PATH_KEY].append(vid_file_path)
                df[SPLIT_KEY].append(splits[pid])
                df[SAMPLE_ID_KEY].append(os.path.basename(vid_file_path).split('.')[0])
    df = pd.DataFrame(df)
    return df


def kth_action_video_nobbox(kth_action_home, clsname_map_path, **kwargs):
    df = kth_action_recg_splits_df(kth_action_home, clsname_map_path)
    for split in df[SPLIT_KEY].unique():
        df_split = df[df[SPLIT_KEY] == split]
        df_split.to_csv(KTH_SPLIT_CSV.format(split=split), index=False)
