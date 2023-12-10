
"""
Official Actor and Action Video Segmentation (A2D) from a Sentence dataset website
https://web.eecs.umich.edu/~jjcorso/r/a2d/

Note: A2DSentences and JHMDB-Sentences are created by providing the additional textual annotations on the original A2D
[45] and JHMDB [16] datasets. A2D-Sentences contain
See: https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Language_As_Queries_for_Referring_Video_Object_Segmentation_CVPR_2022_paper.pdf

Original dataset
https://web.eecs.umich.edu/~jjcorso/r/a2d/index.html#downloads
Extended dataset (Mostly used)
https://kgavrilyuk.github.io/publication/actor_action/
We have extended Actor and Action (A2D) Dataset with additional description of every object is
doing in the videos. We provide three files containing our annotation:

1. a2d_annotation.txt contains annotation in the format “video_id,instance_id,query” where:
https://kgavrilyuk.github.io/actor_action/a2d_annotation.txt

“video_id” - the original id of the video from the A2D dataset
“instance_id” - the id of the object in the video that we have added to the original annotation
“query” - the description of what object is doing throughout the whole video (see the paper for more details)

2. a2d_annotation_with_instances.zip - the original annotation from the A2D dataset in HDF5 with the field
“instance” added. This field corresponds to “instance_id” field in the a2d_annotation.txt file.
https://drive.google.com/file/d/14DNamenZsvZnb32NFBNkZCGene5D2oaE/view

3. a2d_missed_videos.txt contains all the videos that were not annotated with descriptions and
therefore were excluded from experiments in the paper.
https://kgavrilyuk.github.io/actor_action/a2d_missed_videos.txt
"""

import argparse
from utils import download_utils


VID_DATA_URL = 'https://web.eecs.umich.edu/~jjcorso/bigshare/A2D_main_1_0.tar.bz'
METADATA_URL = 'https://web.eecs.umich.edu/~jjcorso/r/a2d/files/README'
EXT_METADATA_URL = 'https://kgavrilyuk.github.io/actor_action/a2d_annotation.txt'
EXT_ANNOTATION_URL = 'https://drive.google.com/file/d/14DNamenZsvZnb32NFBNkZCGene5D2oaE/view?usp=sharing'
EXCLUDE_INST_URL = 'https://kgavrilyuk.github.io/actor_action/a2d_missed_videos.txt'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for Actor and Action Video (A2D) dataset')
    args = parser.parse_args()
    #download_utils.metadata_txt(url=METADATA_URL, dataset_dir=args.dataset_dir)
    download_utils.metadata_txt(url=EXT_METADATA_URL, dataset_dir=args.dataset_dir)
    download_utils.zip(url=VID_DATA_URL, dataset_dir=args.dataset_dir, type='bz')
    download_utils.zip(
        url=EXT_ANNOTATION_URL,
        dataset_dir=args.dataset_dir,
        type='zip',
        output_fname='a2d_inst')