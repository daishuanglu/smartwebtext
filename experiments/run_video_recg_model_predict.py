import argparse
import pandas as pd
import os
from tqdm import tqdm

from utils import train_utils, video_utils
from models import video_recognition
from preprocessors import pipelines


def main():
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', required=True,
        help='model train configuration path.')
    parser.add_argument(
        '--eval_key', required = True,
        help = 'evaluation dataset csv key in the config.')
    parser.add_argument(
        '--model_obj_name', required = True,
        help='video recognition model object class name, must be one of "EncodedSparseCoding",
        "VivitRecgModel", "VitRecgModel", "SwinRecgModel"')
    args = parser.parse_args()
    """
    # For local debugging
    class args:
        config = "config/swin_sc_hm.yaml"
        eval_key = 'eval_hmdb51'
        #model_obj_name = 'VitRecgModel'
        model_obj_name = 'EncodedSparseCoding'
    config = train_utils.read_config(args.config)
    model_obj = getattr(video_recognition, args.model_obj_name, None)
    model = model_obj(**dict(config=config,
                             video_key=pipelines.VIDEO_KEY,
                             target_key=pipelines.CLASS_ID_KEY,
                             num_classes=config[args.eval_key]['num_classes'])
                      )
    if config.get('logger_dir', ''):
        latest_ckpt_path = train_utils.latest_ckpt(config['logger_dir'], config['model_name'])
        model = train_utils.load(model, latest_ckpt_path, strict=False)
    df_eval = pd.read_csv(config[args.eval_key]['path'])
    eval_idx = (df_eval[pipelines.SPLIT_KEY] == 'test') | (df_eval[pipelines.SPLIT_KEY] == 'eval')
    df_eval = df_eval[eval_idx].set_index(pipelines.SAMPLE_ID_KEY)
    if config.get('shuffle', False):
        df_eval = df_eval.sample(frac=1)
    output_fpath = config[args.eval_key]['output_fpath'] % config['model_name']
    #output_fpath = 'evaluation/ucf_recg/ucf_blended_cam/%s/{vid}.mp4' % config['model_name']
    os.makedirs(os.path.dirname(output_fpath), exist_ok=True)
    for vid, row in tqdm(df_eval.iterrows(), total=len(df_eval)):
        viz, _, _ = video_recognition.blended_cam(model=model,
                                                  clip_path=row[pipelines.CLIP_PATH_KEY],
                                                  clip_len=config['clip_len'],
                                                  frame_sample_rate=config['frame_sample_rate'],
                                                  cubelet_size=config['cubelet_size'])
        video_utils.save3d(output_fpath.format(vid=vid), viz)


if __name__ == '__main__':
    main()