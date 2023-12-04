import pandas as pd
from utils import train_utils
from preprocessors import pipelines
from tqdm import tqdm

SPLITS = ['train', 'val', 'test']

def images_to_video():
    # TODO: wrap segmented images to mp4 video
    return

def SAM_video(vid_path, output_path):
    # TODO: SAM segmentation here
    images_to_video(images_path)
    return


def main():
    config = train_utils.read_config("config/a2d_video_segmentation.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.a2d_video_images(
            config['dataset_dir'], config['label_colors_json'], config['train_val_ratio'])

    for split in SPLITS:
        split_path = pipelines.A2D_IMAGE_SPLIT_CSV.format(split= split)
        df = pd.read_csv(split_path)
        for vid in tqdm(df['vid'], desc=split):
            mp4_clip_path = pipelines.A2D_CLIP_PATH.format(vid=vid)
            sam_mp4_clip_output_path = pipelines.A2D_SAM_OUTPUT_CLIP_PATH.format(vid=vid)
            SAM_video(mp4_clip_path, sam_mp4_clip_output_path)



