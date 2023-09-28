import os
import glob
import pandas as pd
import numpy as np

from utils import color_utils


A2D_METADATA_COLUMNS = [
    'vid', 'actor_action_label',
    'start_time', 'end_time', 'height', 'width', 'num_frames', 'num_labeled_frames', 'split_no']
A2D_METADATA_PATH = '{root}/A2D_main_1_0/Release/videoset.csv'
A2D_CLIP_PATH = '{root}/A2D_main_1_0/Release/clips320H/{vid}.mp4'
A2D_SAM_OUTPUT_CLIP_PATH = '{root}/A2D_main_1_0/Release/clips320H_sam/{vid}.mp4'
A2D_ANNOTATION_PATH = '{root}/A2D_main_1_0/Release/Annotations/mat/{vid}/{fid}.mat'
A2D_ANNO_COLOR_MAP_PATH ='{root}/A2D_main_1_0/Release/Annotations/col/{vid}/{fid}.png'
A2D_IMAGE_SPLIT_CSV = 'data_model/A2D_video_image_{split}.csv'
A2D_FID_SEP = ';;;'
A2D_INFO_PATH = '{root}/A2D_main_1_0/Release/README'


def a2d_annotated_frame_ids(root, vid):
    fids = []
    for anno_path in glob.glob(A2D_ANNOTATION_PATH.format(root=root, vid=vid, fid='*')):
        fid = os.path.basename(anno_path).replace('.mat', '')
        fids.append(fid)
    return A2D_FID_SEP.join(fids)


def a2d_video_images(dataset_dir, label_colors_json, train_val_ratio=[0.95, 0.05]):
    df_meta = pd.read_csv(
        A2D_METADATA_PATH.format(root=dataset_dir), header=None,
        dtype=str, na_values=[], parse_dates=False, keep_default_na=False)
    os.makedirs(os.path.dirname(A2D_SAM_OUTPUT_CLIP_PATH), exist_ok=True)
    df_meta.columns = A2D_METADATA_COLUMNS
    df_meta['split'] = df_meta['split_no'].apply(lambda x: np.random.choice(
        ['train', 'val'], 1, p=train_val_ratio, replace=True)[0] if x == '0' else 'test')
    df_meta['fids'] = df_meta['vid'].apply(lambda x: a2d_annotated_frame_ids(dataset_dir, x))
    with open(A2D_INFO_PATH.format(root=dataset_dir), 'r') as f:
        print('A2D color codes:')
        found = False
        label_colors = []
        for line in f.readlines():
            if found:
                name, iid, valid, R, G, B = line.rstrip().split()
                label_color = color_utils.ColorCode(name=name, id=int(iid), color=(int(R),int(G),int(B)))
                label_colors.append(label_color)
                print(label_color)
            found = found or all([kw in line for kw in ['NAME', 'ID', 'Valid','R','G','B']])
    color_utils.save_color_codes(label_colors, label_colors_json)
    for split in df_meta['split'].unique():
        df_split = df_meta[df_meta['split'] == split]
        df_split.drop(columns=['split'])
        df_split.to_csv(A2D_IMAGE_SPLIT_CSV.format(split=split), index=False)


