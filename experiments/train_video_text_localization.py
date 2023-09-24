import random
import cv2
import os

import numpy as np
import pandas as pd
import pims
from tqdm import tqdm
from preprocessors import pipelines, sample_generator
from utils import train_utils, data_utils, video_utils
from models import video_text


def load_ucf_frames(feature_dict, clip_len, frame_sample_rate):
    vf = pims.Video(feature_dict[pipelines.CLIP_PATH_KEY])
    if feature_dict[pipelines.FRAME_ID_KEY]:
        vf = [vf[i] for i in feature_dict[pipelines.FRAME_ID_KEY]]
    indices = video_utils.sample_frame_indices(
        clip_len=clip_len, frame_sample_rate=frame_sample_rate, seg_len=len(vf))
    frames = [np.array(vf[i]) for i in indices]
    return frames


def negative_text(feature_dict):
    rand_sample = {pipelines.TEXT_KEY: feature_dict[pipelines.TEXT_KEY],
                   pipelines.TARGET_KEY: 1.0}
    pos = random.random() > 0.5
    rand_sample[pipelines.TARGET_KEY] = float(pos)
    if not pos:
        rand_sample[pipelines.TEXT_KEY] = sample_generator.rand_line_from_file(
            rand_sample[pipelines.TEXT_KEY], pipelines.VIDTXT_ALL_TEXTS, max_retry=3)
    return rand_sample


def main():
    class args:
        config = "config/video_text_cam_mixed.yaml"
        #config = "config/video_text_cam_ucf.yaml"
    config = train_utils.read_config(args.config)
    if not config.get("skip_prep_data", False):
        pipelines.mixed_video_text(config['datasets'])
    train_ucf_recg_features = {
        pipelines.CLIP_PATH_KEY: (lambda x: str(x)),
        pipelines.TEXT_KEY: (lambda x: str(x)),
        pipelines.CLASS_NAME: (lambda x: str(x)),
        pipelines.SAMPLE_ID_KEY: lambda x: str(x),
        pipelines.FRAME_ID_KEY: lambda x: [int(i) for i in str(x).strip().split(pipelines.FRAME_ID_SEP) if i]
    }
    train_ucf_recg_cols = {
        pipelines.VIDEO_KEY: lambda x: load_ucf_frames(
            x,
            config['clip_len'],
            config['frame_sample_rate']),
        'sampled_targets': (lambda x: negative_text(x))
    }
    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    os.makedirs(logger_dir, exist_ok=True)
    model_obj = video_text.VideoTextDiscLocalization(
        config,
        video_key=pipelines.VIDEO_KEY,
        target_key=pipelines.TARGET_KEY,
        text_key=pipelines.TEXT_KEY)
    print("model initialized. ")
    model_obj = model_obj.to(train_utils.device)
    print("model moved to device: ", train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
        if config['resume_ckpt'] else None
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            pipelines.VIDTXT_TRAIN_SPLIT_CSV.format(split='train'),
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
            pipelines.VIDTXT_TRAIN_SPLIT_CSV.format(split='val'),
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
    output_vid_fpath = 'evaluation/vid_text/%s/blended_cam/{vid}.mp4' % config['model_name']
    output_txt_fpath = 'evaluation/vid_text/%s/text_cam/{vid}.jpg' % config['model_name']
    os.makedirs(os.path.dirname(output_vid_fpath), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt_fpath), exist_ok=True)
    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        viz_vid, viz_text = model.blended_video_text_cam(row[pipelines.CLIP_PATH_KEY],
                                                         row[pipelines.TEXT_KEY])
        vid = os.path.basename(row[pipelines.CLIP_PATH_KEY]).split('.')[0]
        video_utils.save3d(output_vid_fpath.format(vid=vid), viz_vid)
        cv2.imwrite(output_txt_fpath.format(vid=vid), viz_text)

if __name__ == '__main__':
    main()