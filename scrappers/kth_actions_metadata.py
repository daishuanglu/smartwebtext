"""
Scrapper for extract KTH action videos. example usage
python3 scrappers/kth_action_videos.py\
    --output_dir=path/to/output/folder\
    --dataset_dir=path/to/your/kth/action/dataset\
    --limit=None

Running this script will require you to first download the KTH action videos from
https://www.csc.kth.se/cvap/actions, unzip each set of action videos as a folder
http://www.csc.kth.se/cvap/actions/walking.zip
http://www.csc.kth.se/cvap/actions/jogging.zip
http://www.csc.kth.se/cvap/actions/running.zip
http://www.csc.kth.se/cvap/actions/boxing.zip
http://www.csc.kth.se/cvap/actions/handwaving.zip
http://www.csc.kth.se/cvap/actions/handclapping.zip
and the sequence info file https://www.csc.kth.se/cvap/actions/00sequences.txt
and save into a single directory.
"""

import argparse
import os
import pims
import pandas as pd
from transformers import pipeline
from PIL import Image
from tqdm import tqdm
import hashlib
import uuid
import torch

accelerator, device = ("gpu", "cuda:0") if torch.cuda.is_available() else ("cpu", "cpu")

OBJ_DETECTOR = pipeline(model="facebook/detr-resnet-50", device=device)
#ZS_OBJ_DETECTOR = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")

# download and unzip KTH action videos from here https://www.csc.kth.se/cvap/actions/
#KTH_ACTION_HOME = 'C:\\Users\\shud0\\KTHactions'
#KTH_ACTION_HOME = '/home/shuangludai/KTHactions'
KTH_VIDEO_FILE = '{action}/person{pid}_{action}_{var}_uncomp.avi'
KTH_ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

def generate_uuid_from_string(the_string):
    md5_hash = hashlib.md5()
    md5_hash.update(the_string.encode("utf-8"))
    the_md5_hex_str = md5_hash.hexdigest()
    return str(uuid.UUID(the_md5_hex_str))


def get_synthetic_bboxes(vid_file_path, fids, labels=['person']):
    v = pims.Video(vid_file_path)
    df_data = pd.DataFrame([], columns=['fid', 'x', 'y', 'w', 'h', 'confidence'], index=fids)
    for fid in tqdm(fids, desc='frame'):
        _fid = int(fid)
        #print(fid, len(v))
        if _fid >= len(v):
            break
        img = Image.fromarray(v[_fid])
        #results = ZS_OBJ_DETECTOR(img, candidate_labels=labels)
        results = OBJ_DETECTOR(img)
        results = [res for res in results if res['label'] in labels]
        if not results:
            continue
        for res in results:
            df_data.at[fid, 'fid'] = _fid
            x,y,w,h = res['box']['xmin'], res['box']['ymin'],\
                res['box']['xmax']-res['box']['xmin'], res['box']['ymax']-res['box']['ymin']
            df_data.at[fid, 'x'] = x / img.width
            df_data.at[fid, 'y'] = y / img.height
            df_data.at[fid, 'w'] = w / img.width
            df_data.at[fid, 'h'] = h / img.height
            df_data.at[fid, 'height'] = img.height
            df_data.at[fid, 'width'] = img.width
            df_data.at[fid, 'confidence'] = res['score']
    return df_data.reset_index()


def parse_kth_splits(kth_action_home, limit=10**10):
    print('dataset_home_dir=', kth_action_home, ' limit=', str(limit))
    fpath = os.path.join(kth_action_home, '00sequences.txt')
    splits = {}
    dfs = []
    n = 0
    with open(fpath, 'r') as f:
        for line in tqdm(f.readlines(), desc='KTH bbox synthetic data'):
            if n >= limit:
                break
            if 'training:' in line.lower():
                splits.update({pid: 'train' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'validation:' in line.lower():
                splits.update({pid: 'val' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'test:' in line.lower():
                splits.update({pid: 'val' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'frames' in line.lower():
                vid_file, kfs = line.strip().split('frames')
                vid_file = vid_file.strip()
                #fids = sum([frange.split('-') for frange in kfs.strip().split(', ')], [])
                #fids = [int(id) - 1 for id in fids]
                fid_ranges = [frange.split('-') for frange in kfs.strip().split(', ')]
                fids = []
                for start, end in fid_ranges:
                    fids += list(range(int(start), int(end)))
                pid, action, var = vid_file.split('_')
                pid = pid.split('person')[1]
                for var in ['d1', 'd2', 'd3', 'd4']:
                    vid_file_basepath = KTH_VIDEO_FILE.format(action=action, pid=pid, var=var)
                    vid_file_path = os.path.join(
                        kth_action_home, vid_file_basepath)
                    if os.path.exists(vid_file_path):
                        print(vid_file_path)
                        df_bbox = get_synthetic_bboxes(vid_file_path, fids, labels=['person'])
                        df_bbox['pid'] = pid
                        df_bbox['action'] = action
                        df_bbox['split'] = splits[pid]
                        df_bbox['video_path'] = vid_file_basepath
                        df_bbox['vid'] = generate_uuid_from_string(vid_file_path)
                        dfs.append(df_bbox)
                        n += 1

    df_data = pd.concat(dfs)
    return df_data


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory for the extracted kth actions.csv file')
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Path to the downloaded KTH action video dataset local directory'
        'download from https://www.csc.kth.se/cvap/actions/ and unzip action videos into a directory')
    parser.add_argument(
        '--limit', required=False, type=int, default=10**10,
        help='limit the number of video to process for debugging.')
    args = parser.parse_args()

    kth_data = parse_kth_splits(kth_action_home=args.dataset_dir, limit=args.limit)
    os.makedirs('data_model', exist_ok=True)
    kth_data.to_csv(os.path.join(args.output_dir, 'kth_actions.csv'), index=False)
