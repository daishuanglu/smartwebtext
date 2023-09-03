"""
UCF101, action recognition dataset
https://www.crcv.ucf.edu/research/data-sets/ucf101/

Publication
Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, UCF101: A Dataset of 101 Human Action Classes From Videos
in The Wild, CRCV-TR-12-01, November, 2012.

Overview
UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101
action categories. This data set is an extension of UCF50 data set which has 50 action categories.

With 13320 videos from 101 action categories, UCF101 gives the largest diversity in terms of actions and
with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint,
cluttered background, illumination conditions, etc, it is the most challenging data set to date. As most of
the available action recognition data sets are not realistic and are staged by actors, UCF101 aims to encourage
further research into action recognition by learning and exploring new realistic action categories.

Data Set Details
The videos in 101 action categories are grouped into 25 groups, where each group can consist of 4-7 videos of
an action. The videos from the same group may share some common features, such as similar background, similar
viewpoint, etc.

The action categories can be divided into five types:
Human-Object Interaction
Body-Motion Only
Human-Human Interaction
Playing Musical Instruments
Sports
"""

import argparse
import ssl

from utils import download_utils

ssl._create_default_https_context = ssl._create_unverified_context
DATASET_URL = 'https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for UCF101 dataset')
    args = parser.parse_args()
    #class args:
    #    dataset_dir = 'D:/video_datasets'
    sav_dir = download_utils.zip(url=DATASET_URL, dataset_dir=args.dataset_dir, type='rar')
    print('UCF 101 dataset saved at ', sav_dir)