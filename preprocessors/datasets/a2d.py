import os
import glob
import pandas as pd
import numpy as np
import json

from preprocessors.constants import *


A2D_METADATA_COLUMNS = [
    'vid', 'actor_action_label',
    'start_time', 'end_time', 'height', 'width', 'num_frames', 'num_labeled_frames', 'split_no']
A2D_METADATA_PATH = '{root}/A2D_main_1_0/Release/videoset.csv'
A2D_CLIP_PATH = '{root}/A2D_main_1_0/Release/clips320H/{vid}.mp4'
A2D_SAM_OUTPUT_CLIP_PATH = '{root}/A2D_main_1_0/Release/clips320H_sam/{vid}.mp4'
#A2D_ANNOTATION_PATH = '{root}/A2D_main_1_0/Release/Annotations/mat/{vid}/{fid}.mat'
A2D_ANNOTATION_PATH = '{root}/a2d_annotation_with_instances/{vid}/{fid}.h5'
A2D_ANNO_COLOR_IMG_PATH ='{root}/A2D_main_1_0/Release/Annotations/col/{vid}/{fid}.png'
A2D_IMAGE_SPLIT_CSV = 'data_model/A2D_video_image_{split}.csv'
A2D_INFO_PATH = '{root}/A2D_main_1_0/Release/README'
A2D_RECG_TRAIN_SPLIT_CSV = 'data_model/A2D_video_recognition_{split}.csv'
A2D_REF_TEXT_PATH = '{root}/A2D_main_1_0/a2d_annotation.txt'
A2D_OBJECT_MASK_KEY = 'reMask'
A2D_LABEL_MASK_KEY = 'reS_id' 
A2D_REF_TEXT_KEY = 'instance'
A2D_PROCESSOR = 'a2d_context'


def root_from_annotation_path(fpath):
    assert 'a2d_annotation_with_instances' in fpath, \
        f'Extract vid from annotation path error: {fpath} is not an A2D annotation path.'
    return fpath.split('/a2d_annotation_with_instances/')[0]


def vid_from_annotation_path(fpath):
    assert 'a2d_annotation_with_instances' in fpath, \
        f'Extract vid from annotation path error: {fpath} is not an A2D annotation path.'
    return os.path.basename(os.path.dirname(fpath))


def load_ref_text_df(root):
    df = pd.read_csv(A2D_REF_TEXT_PATH.format(root=root))
    df['instance_id'] = df['instance_id'].apply(lambda x: int(str(x).split()[0]))
    return df


def get_ref_text(root, vid, obj_id, df=None):
    if df is None:
        df = load_ref_text_df(root=root)
    text = df[(df['instance_id'] == obj_id) & (df['video_id'] == vid)]['query']
    return text.iloc[0]


def a2d_annotated_frame_ids(root, vid):
    fids = []
    fpaths = []
    for anno_path in glob.glob(A2D_ANNOTATION_PATH.format(root=root, vid=vid, fid='*')):
        fid = os.path.basename(anno_path).replace('.h5', '')
        fids.append(fid)
        fpaths.append(anno_path)
    return FRAME_ID_SEP.join(fids), FRAME_ID_SEP.join(fpaths)


def a2d_splits_df(dataset_dir, train_val_ratio=[0.95, 0.05], **kwargs):
    df_meta = pd.read_csv(
        A2D_METADATA_PATH.format(root=dataset_dir), header=None,
        dtype=str, na_values=[], parse_dates=False, keep_default_na=False)
    df_meta.columns = A2D_METADATA_COLUMNS
    df_meta[SPLIT_KEY] = df_meta['split_no'].apply(lambda x: np.random.choice(
        ['train', 'val'], 1, p=train_val_ratio, replace=True)[0] if x == '0' else 'test')
    with open(A2D_INFO_PATH.format(root=dataset_dir), 'r') as f:
        print('A2D color codes:')
        found = False
        aa_id = {i: '' for i in range(80)}
        label_colors = []
        for line in f.readlines():
            if found:
                name, iid, valid, R, G, B = line.rstrip().split()
                label_colors.append({'name': name, 
                                     'id': int(iid), 
                                     'R': int(R), 
                                     'G': int(G),
                                     'B': int(B)})
                aa_id[int(iid)] = name
            found = found or all([kw in line for kw in ['NAME', 'ID', 'Valid','R','G','B']])
        print(label_colors)
    df_meta[CLIP_PATH_KEY] = df_meta['vid'].apply(
        lambda x: A2D_CLIP_PATH.format(root=dataset_dir, vid=x))
    df_meta[[ANNOTATED_FRAME_ID, ANNOTATED_FRAME_PATH]] = df_meta['vid'].apply(
        lambda x: a2d_annotated_frame_ids(dataset_dir, x)).apply(pd.Series)
    df_meta[CLASS_ID_KEY] = df_meta['actor_action_label']
    df_meta[CLASS_NAME] = df_meta[CLASS_ID_KEY].astype(int).map(aa_id)
    df_meta[['actor', 'action']] =  df_meta[CLASS_NAME].str.split('-', n=2, expand=True)
    df_meta[SAMPLE_ID_KEY] = df_meta['vid']
    df_meta[CLASS_ID_NAME_MAP] = json.dumps(aa_id)
    df_meta[COLOR_CODE] = json.dumps(label_colors)
    df_meta[COLOR_CODE] = json.dumps(label_colors)
    df_meta[OBJECT_MASK_KEY] = A2D_OBJECT_MASK_KEY
    df_meta[LABEL_MASK_KEY] = A2D_LABEL_MASK_KEY
    df_meta[REF_TEXT_KEY] = A2D_REF_TEXT_KEY
    df_meta[ANNOTATION_PROCESSOR] = A2D_PROCESSOR
    return df_meta


if __name__ == '__main__':
    dataset_dir = "D:/video_datasets"
    df = a2d_splits_df(dataset_dir, train_val_ratio=[0.95, 0.05])
    print(df.head(3))