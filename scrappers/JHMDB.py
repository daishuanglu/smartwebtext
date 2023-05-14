"""
J-HMDB Sentences
Official Joint-annotated Human Motion Data Base website
A fully annotated data set for human actions and human poses.
http://jhmdb.is.tue.mpg.de/

Note: A2DSentences and JHMDB-Sentences are created by providing the additional textual annotations on the original A2D
[45] and JHMDB [16] datasets. A2D-Sentences contain
See: https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Language_As_Queries_for_Referring_Video_Object_Segmentation_CVPR_2022_paper.pdf

Original dataset
https://web.eecs.umich.edu/~jjcorso/r/a2d/index.html#downloads

Download
1. VIDEO DATABASE
HMDB51 – About 2GB for a total of 7,000 clips distributed in 51 action classes.
Stabilized HMDB51 – the number of clips and classes are the same as HMDB51, but there is a mask in [video_name].form associated with each clip. The mask file is readable in matlab.
README
bounding boxes (link to INRIA)
2. HOG/HOF (STIP) FEATURES
STIP features for the HMDB51 – ~3.5GB, (for the binaries see http://www.irisa.fr/vista/Equipe/People/Laptev/download.html )
STIP features for the stabilized HMDB51 – ~ 2.9GB
3. THREE SPLITS
three splits for the HMDB51
README

Extended JHMDB dataset (Mostly used)
https://kgavrilyuk.github.io/publication/actor_action/

We have extended J-HMDB Dataset with additional description of every human is doing in the videos:

1. jhmdb_annotation.txt contains annotation in the format “video_id,query”:
https://kgavrilyuk.github.io/actor_action/jhmdb_annotation.txt
“video_id” - the original id of the video from the J-HMDB dataset
“query” - the description of what human is doing throughout the whole video (see the paper for more details)
"""

import argparse
import glob
import os

from utils import download_utils

# Maybe only need to use the STABLE videos with its masks
STABLE_VID_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_sta.rar'
# mask_out -  a [frame height] x [frame width] x [ # frames] matrix,
# the mask image for every frame (only pixels with 1 are valid, the rest is background)
METADATA_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/stabilized_readme.txt'
EXT_METADATA_URL = 'https://kgavrilyuk.github.io/actor_action/jhmdb_annotation.txt'
SPLITS_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'
SPLITS_METADATA_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/split_readme.txt'

# URLs may not be used including SIFT features, unstablized dataset, etc.
VID_DATA_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
SIFT_FEATURE_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_sta_stips.rar'
STABLE_SIFT_FEATURE_URL = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_sta_stips.rar'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for Joint HMDB dataset')
    args = parser.parse_args()
    download_utils.metadata(url=METADATA_URL, dataset_dir=args.dataset_dir)
    download_utils.metadata(url=EXT_METADATA_URL, dataset_dir=args.dataset_dir)
    download_utils.metadata(url=SPLITS_METADATA_URL, dataset_dir=args.dataset_dir)
    download_utils.zip(url=SPLITS_URL, dataset_dir=args.dataset_dir, type='rar')
    download_utils.zip(url=STABLE_VID_URL, dataset_dir=args.dataset_dir, type='rar')
    video_data_dir = os.path.join(args.dataset_dir, STABLE_VID_URL.split('/')[-1].split('.')[0])
    for fname in glob.glob(os.path.join(video_data_dir, '*.rar')):
        print('Unzipping: ', fname)
        download_utils.unzip_file(fname, video_data_dir, 'rar')
        print('Delete: ', fname)
        os.remove(fname)
