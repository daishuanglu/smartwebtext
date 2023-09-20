import argparse
import os
import random

import numpy as np
import pandas as pd
import pims
from tqdm import tqdm
from preprocessors import pipelines
from utils import train_utils, data_utils, video_utils
from models import video_recognition


def load_ucf_frames(feature_dict,
                    clip_len,
                    frame_sample_rate,
                    zoom_in_scale=1,
                    rand_crop_rate=0.0):
    vf = pims.Video(feature_dict['clip_path'])
    indices = video_utils.sample_frame_indices(
        clip_len=clip_len, frame_sample_rate=frame_sample_rate, seg_len=len(vf))
    frames = [np.array(vf[i]) for i in indices]
    if zoom_in_scale > 1 and (rand_crop_rate > 0.0):
        frame_size = frames[0].shape
        crop_size = (frame_size[0] // zoom_in_scale,
                     frame_size[1] // zoom_in_scale)
        if random.random() > (1 - rand_crop_rate):
            frames = video_utils.random_crop_frames_np(frames, crop_size)
    return frames


def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    '--config', required=True,
    #    help='model training configuration path.')
    #args = parser.parse_args()
    # For local debugging
    class args:
        config = "config/vivit_cam_ucf_recg.yaml"
    config = train_utils.read_config(args.config)
    if not config.get("skip_prep_data", False):
        pipelines.ucf_recognition(config['dataset_dir'], config['train_val_ratio'])
    train_ucf_recg_features = {
        'vid_path': (lambda x: str(x)),
        pipelines.UCF_CLIP_PATH: (lambda x: str(x)),
        pipelines.UCF_CLASS_IDX: (lambda x: int(x) - 1),
        'className': (lambda x: str(x))
    }
    train_ucf_recg_cols = {
        pipelines.UCF_VIDEO: lambda x: load_ucf_frames(
            x,
            config['clip_len'],
            config['frame_sample_rate'],
            config.get('zoom_in_scale', 1),
            config.get('rand_crop_rate', 0.0)),
        'id': lambda x: os.path.basename(x['vid_path']).split('.')[0]
    }
    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    os.makedirs(logger_dir, exist_ok=True)
    model_obj = video_recognition.Vivit(
        config,
        video_key=pipelines.UCF_VIDEO,
        target_key=pipelines.UCF_CLASS_IDX,
        num_classes=pipelines.UCF_NUM_CLASSES)
    print("model initialized. ")
    model_obj = model_obj.to(train_utils.device)
    print("model moved to device: ", train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
        if config['resume_ckpt'] else None
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            pipelines.UCF_RECG_TRAIN_SPLIT_CSV.format(split='train'),
            train_ucf_recg_features,
            col_fns=train_ucf_recg_cols,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            flat_cols=True,
            shuffle=True,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        val_dl = data_utils.get_context_csv_data_loader(
            pipelines.UCF_RECG_TRAIN_SPLIT_CSV.format(split='val'),
            train_ucf_recg_features,
            col_fns=train_ucf_recg_cols,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            flat_cols=True,
            shuffle=False,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        print("start training, logging at %s" % logger_dir)
        model_obj, _ = train_utils.training_pipeline(
            model_obj,
            train_dl,
            val_x=val_dl,
            nepochs=config['epochs'],
            resume_ckpt=latest_ckpt_path,
            model_name=config['model_name'],
            monitor=config['monitor'],
            logger_path=logger_dir)
    print('-----------------  done training ----------------------------')
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name'])
    model = train_utils.load(model_obj, latest_ckpt_path, strict=False)
    df_eval = pd.read_csv(pipelines.UCF_RECG_TRAIN_SPLIT_CSV.format(split='val'))
    output_fpath = 'evaluation/ucf_recg/ucf_blended_cam/%s/{vid}.mp4' % config['model_name']
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        viz = model.blended_cam(row[pipelines.UCF_CLIP_PATH])
        vid = os.path.basename(row[pipelines.UCF_CLIP_PATH]).split('.')[0]
        video_utils.save3d(output_fpath.format(vid=vid), viz)


if __name__ == '__main__':
    main()