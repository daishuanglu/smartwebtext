"""
Refer-YouTube-VOS dataset
Referring Video Object Segmentation
https://youtube-vos.org/dataset/rvos/

Referring video object segmentation aims at segmenting an object in video with language expressions.
Unlike the previous video object segmentation, the task exploits a different type of supervision,
language expressions, to identify and segment an object referred by the given language expressions in a video.
A detailed explanation of the new task can be found in the following paper.
See: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600205.pdf

Important Announcement: Unlike prior Youtube-VOS challenges, the Refer-Youtube-VOS challenge winners
will be required to release the experimental codes to replicate their results.
The detailed procedure for releasing the code is to be determined.

Dataset StatisticsPermalink
We collected the large-scale dataset for referring video object segmentation, called Refer-YouTube-VOS,
which is based on YouTube-VOS-2019 dataset. Specifically, our new dataset has

3,978 high-resolution YouTube videos
131k high-quality manual annotations
15k language expressions
We split the YouTube-VOS dataset into 3,471 training videos, 202 validation videos and 305 test videos.
Note that the splits have changed from Youtube-VOS-2019.

EvaluationPermalink
Following the previous video object segmentation challenge Youtube-VOS, we use Region Jaccard (J)
and Boundary F measure (F) as our evaluation metrics. We compute J and F, averaged over all corresponding
expressions.

We have set up evaluation servers on CodaLab for the convenience of evaluating new algorithms.
For more details of how to submit your results, please check the following links.
https://competitions.codalab.org/competitions/29139

Note: The training file is using the 2019 competition and need the meta json files in meta_expressions.zip

In training dataset, each video is annotated with one or multiple objects.
The original frames are stored in the "JPEGImages" folder and the object masks are stored in the "Annotations"
 folder. A meta json file describing the data information with language expressions is provided in
 meta_expressions.zip file.

train.zip
    |- JPEGImages
        |- <video_id>
            |- <frame_id>.jpg
            |- <frame_id>.jpg
        |- <video_id>
            |- <frame_id>.jpg
            |- <frame_id>.jpg
    |- Annotations
        |- <video_id>
            |- <frame_id>.png
            |- <frame_id>.png
        |- <video_id>
            |- <frame_id>.png
            |- <frame_id>.png

meta_expressions.json
    {
        "videos": {
            "<video_id>": {
                "expressions": {
                    "<expression_id>": {
                        "exp": "<expression>",
                        "obj_id": "<object_id>"
                    }
                },
                "frames": [
                    "<frame_id>",
                    "<frame_id>"
                    ]
                }
            }
        }
    }
# <object_id> is the same as the pixel values of object in annotated segmentation PNG files.
# <frame_id> is the 5-digit index of frame in video, and not necessary to start from 0.

For submission, only the annotations are necessary. Note that the submission format and folder structure
are different from the provided training set, where it requires one folder for each expression, storing that
the referred object's binary masks. All the frames in meta file of validation/test should be submitted.
Please follow the format below.

submission.zip
        |- Annotations
            |- <video_0>
                |- <expression_0>
                    |- <frame_0>.png
                    |- <frame_1>.png
            |- <video_1>
                |- <expression_0>
                    |- <frame_0>.png
                    |- <frame_1>.png

There is an example "valid_submission_sample.zip" in download links above. The example is the provided
annotated segmentations of random videos from the training set.
"""

import argparse
from utils import download_utils

# Dataset google drive share folder link
MAIN2021_URL = 'https://drive.google.com/drive/folders/1J45ubR8Y24wQ6dzKOTkfpd9GS_F9A2kb'
MAIN2019_URL = 'https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz'

# Dataset links. Note: the 2021 competition still used the 2019 training set.
TRAIN2019_URL = 'https://drive.google.com/file/d/13Eqw0gVK-AO5B-cqvJ203mZ2vzWck9s4/view?usp=share_link'
VAL2021_URL = 'https://drive.google.com/file/d/1yH9YywIBzNfepwLLqzxFHXQxa89tjrkq/view?usp=share_link'
EXMAPLE_SUBMISSION_URL = 'https://drive.google.com/file/d/1NRoPk0XalL9V4_SFyXdfP2WVsO6P1QNu/view?usp=share_link'
TRAIN2021_INFO_URL = 'https://drive.google.com/file/d/1BqeRWN52efx4BYQgc5NIppOo0Ox3c4Fi/view?usp=share_link'
TEXT_EXPR_URL = 'https://drive.google.com/file/d/127nwMmSDSqlNx8dryCq1cH8I_OR0kAaM/view?usp=share_link'

# Unused links including test sets, old validation sets, etc.
TEST2019_URL = 'https://drive.google.com/file/d/1S50D-vwOKrmTJNh6VDfXhj8L0jkrNA6V/view?usp=share_link'
VAL2019_URL = 'https://drive.google.com/file/d/1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t/view?usp=share_link'
TEST2021_URL = 'https://drive.google.com/file/d/15Y2rBR0LqMpcTEvBkbUH9h-Ljq8LMYLR/view?usp=share_link'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for Youtube RVOS dataset.')
    args = parser.parse_args()
    download_utils.metadata(
        url=TRAIN2021_INFO_URL, dataset_dir=args.dataset_dir, output_fname='train2021_info.txt')
    download_utils.zip(
        url=TEXT_EXPR_URL, dataset_dir=args.dataset_dir, type='zip', output_fname='train2021_text_expr')
    download_utils.zip(
        url=TRAIN2019_URL, dataset_dir=args.dataset_dir, type='zip', output_fname='train')
    download_utils.zip(
        url=VAL2021_URL, dataset_dir=args.dataset_dir, type='zip', output_fname='valid')