"""
Kinetics 400, 600 and 700 video dataset

A collection of large-scale, high-quality datasets of URL links of up to 650,000 video clips that
cover 400/600/700 human action classes, depending on the dataset version. The videos include human-object
interactions such as playing instruments, as well as human-human interactions such as shaking hands and hugging.
Each action class has at least 400/600/700 video clips. Each clip is human annotated with a single action
class and lasts around 10 seconds.
"""

import argparse
import json
import os
import pandas as pd
from p_tqdm import p_map
from tqdm import tqdm
from utils import download_utils
import cv2
import pims
import shutil
from functools import partial


# Meta_data urls
K400_METADATA_URL = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz'
K600_METADATA_URL = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz'
K700_METADATA_URL = 'https://storage.googleapis.com/deepmind-media/Datasets/kinetics700.tar.gz'

DATASET_URL = {'400': K400_METADATA_URL,
               '600': K600_METADATA_URL,
               '700': K700_METADATA_URL}
SPLITS = ['train', 'test', 'validate']


def dl_video(vid_sample, data_dir, vsize):
    video_path = os.path.join(data_dir, '{vid}.mp4')
    vid, sample = vid_sample
    status = download_utils.download_clip_from_youtube(
        video_url=sample['url'], local_path=video_path.format(vid=vid))
    start_sec, end_sec = sample['annotations']['segment']
    if status['status'] == 'success':
        vf = pims.Video(video_path.format(vid=vid))
        start_fid = int(vf.frame_rate * start_sec)
        end_fid = int(vf.frame_rate * end_sec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_vfname = os.path.basename(video_path.format(vid=vid))
        writer = cv2.VideoWriter(tmp_vfname, fourcc, vf.frame_rate, vsize)
        for i in range(start_fid, min(end_fid, len(vf))):
            resized_frame = cv2.resize(vf[i], vsize)
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            writer.write(resized_frame)
        writer.release()
        shutil.move(tmp_vfname, video_path.format(vid=vid))
        return {'clip_path': '/'.join([sample['subset'], vid]),
                'vid': vid,
                'label': sample['annotations']['label'],
                'split': sample['subset']}
    else:
        return {}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for Kinetics dataset')
    parser.add_argument(
        '--set', required=True,
        help='dataset name one of 400, 600, 700')
    parser.add_argument(
        '--size', default='224,224',
        help='resize video to size height,width.')
    args = parser.parse_args()
    vsize = args.size.split(',')
    vsize = int(vsize[0]), int(vsize[1])
    if args.set not in ['400', '600', '700']:
        print('kinetics set must be one of 400, 600 and 700.')
        exit(0)
    sav_dir = download_utils.zip(url=DATASET_URL[args.set], dataset_dir=args.dataset_dir, type='gz')
    for split in SPLITS:
        json_path = os.path.join('{sav_dir}', '{split}.json').format(sav_dir=sav_dir, split=split)
        data_dir = os.path.join('{sav_dir}', '{split}').format(sav_dir=sav_dir, split=split)
        metadata_path = os.path.join(data_dir, '{split}_split.csv').format(split=split)

        with open(json_path, 'r') as f:
            data = json.load(f)
        os.makedirs(data_dir, exist_ok=True)
        df = p_map(partial(dl_video, data_dir=data_dir, vsize=vsize), list(data.items()))
        df = [d for d in df if d]
        pd.DataFrame(df).to_csv(metadata_path, index=False)