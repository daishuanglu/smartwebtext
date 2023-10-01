"""
HMDB51 dataset
What is HMDB51 Dataset?
The HMDB51 (Human Motion Database 51) dataset is created to enhance the research in computer vision research
of recognition and search in the video. A lot of effort has been put into the collection and annotation of
large scalable static images with large image categories, but a similar effort has not been done in video
division. This dataset has been collected to push the efforts in the collection and annotation of video
datasets. This dataset is a collection of various sources such as movies, and public databases
(Prelinger archive, YouTube, and Google videos). This dataset contains a minimum of 101 clips for
each 51 action categories and in total, the dataset contains 6849 clips.

Download HMDB51 Dataset in Python
Instead of downloading the HMDB51 dataset in Python, you can effortlessly load it in Python via our
Deep Lake open-source with just one line of code.

Load HMDB51 Dataset Training Subset in Python
import deeplake
ds = deeplake.load("hub://activeloop/hmdb51-train")
Load HMDB51 Dataset Testing Subset in Python
import deeplake
ds = deeplake.load("hub://activeloop/hmdb51-test")
Load HMDB51 Dataset Extras Subset in Python
import deeplake
ds = deeplake.load("hub://activeloop/hmdb-extras")
HMDB51 Dataset Structure
HMDB51 Data Fields
videos: tensor containing videos
labels: tensor containing labels for their respective videos
video_quality: tensor containing video quality label
number_of_people: tensor containing a number of people in a video
camera_viewpoint: tensor containing camera viewpoint label for a video
camera_motion: tensor containing camera motion label for a video
visible_body_parts: tensor containing visible body parts label for a video
How to use HMDB51 Dataset with PyTorch and TensorFlow in Python
Train a model on the HMDB51 dataset with PyTorch in Python
Let’s use Deep Lake built-in PyTorch one-line dataloader to connect the data to the compute:

dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
Train a model on the HMDB51 dataset with TensorFlow in Python
dataloader = ds.tensorflow()

There are totally 153 files in this folder,
[action]_test_split[1-3].txt  corresponding to three splits reported in the paper.
The format of each file is
[video_name] [id]
The video is included in the training set if id is 1
The video is included in the testing set if id is 2
The video is not included for training/testing if id is 0
There should be 70 videos with id 1 , 30 videos with id 2 in each txt file.

"""

import argparse
import os
import glob
from utils import download_utils


VIDEO_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
ANNO_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for something-something-v2 dataset')
    args = parser.parse_args()
    #class args:
    #    dataset_dir = 'D:/video_datasets/hmdb51'
    os.makedirs(args.dataset_dir, exist_ok=True)
    fname = download_utils.dl(VIDEO_URL)
    download_utils.unzip_file(fname, args.dataset_dir, type='rar')
    for fpath in glob.glob(os.path.join(args.dataset_dir, '*.rar')):
        download_utils.unzip_file(fpath, args.dataset_dir, type='rar')
    download_utils.zip(ANNO_URL, args.dataset_dir, type='rar')