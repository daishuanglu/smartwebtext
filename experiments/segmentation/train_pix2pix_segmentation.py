import os

import pandas as pd
import torch
from preprocessors.video_annotation import FrameAnnotation
from tqdm import tqdm
from PIL import Image as PilImage
import pims

from preprocessors import pipelines
from utils import train_utils
from utils import data_utils
from models import segmentation


def load_frame_images(feature_dict):
    fids = [int(fid) for fid in feature_dict[
        pipelines.ANNOTATED_FRAME_ID].split(pipelines.FRAME_ID_SEP)]
    vf =feature_dict[pipelines.CLIP_PATH_KEY]
    v = pims.Video(vf)
    samples = [torch.from_numpy(v[i]) for i in fids]
    #samples = [PilImage.fromarray(v[i]) for i in fids]
    return samples


def load_frame_annotation(feature_dict):
    annotation = FrameAnnotation(
        object_mask_key=feature_dict[pipelines.OBJECT_MASK_KEY], 
        label_mask_key=feature_dict[pipelines.LABEL_MASK_KEY], 
        ref_text_key=feature_dict[pipelines.REF_TEXT_KEY],
        processor=feature_dict[pipelines.ANNOTATION_PROCESSOR])
    obj_masks = []
    label_masks = []
    ref_texts = []
    annotation_paths = feature_dict[pipelines.ANNOTATED_FRAME_PATH].split(pipelines.FRAME_ID_SEP)
    for annotation_path in annotation_paths:
        annotation(annotation_path)
        obj_masks.append(torch.from_numpy(annotation.context.obj_mask).long())
        label_masks.append(torch.from_numpy(annotation.context.label_mask).long())
        ref_texts.append(annotation.context.ref_text)
    return {'obj_mask': obj_masks, 'ref_text': ref_texts, 'label_mask': label_masks}


def main():
    config = train_utils.read_config("config/pix2pix_video_segmentation.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.mixed_video_segmentation(config['datasets'])

    train_vid_features = {
        pipelines.OBJECT_MASK_KEY: (lambda x: str(x)), 
        pipelines.LABEL_MASK_KEY: (lambda x: str(x)),
        pipelines.REF_TEXT_KEY: (lambda x: str(x)),
        pipelines.ANNOTATION_PROCESSOR: (lambda x: str(x)),
        pipelines.CLIP_PATH_KEY: (lambda x: str(x)),
        pipelines.ANNOTATED_FRAME_ID: (lambda x: str(x)),
        pipelines.ANNOTATED_FRAME_PATH: (lambda x: str(x)),
        pipelines.SAMPLE_ID_KEY: (lambda x: str(x)),
    }
    train_a2d_cols = {
        'frames': lambda x: load_frame_images(x),
        'gt_frames': lambda x: load_frame_annotation(x)
    }
    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    os.makedirs(logger_dir, exist_ok=True)
    model_obj = segmentation.Hourglass(
        config, multi_fname_sep=pipelines.FRAME_ID_SEP).to(train_utils.device)
    print("model initialized. ")
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
        if config['resume_ckpt'] else None
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            pipelines.VID_SEG_TRAIN_SPLIT_CSV.format(split='train'),
            train_vid_features,
            col_fns=train_a2d_cols,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=pipelines.VID_SEG_DATASET_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        val_dl = data_utils.get_context_csv_data_loader(
            pipelines.VID_SEG_TRAIN_SPLIT_CSV.format(split='val'),
            train_vid_features,
            col_fns=train_a2d_cols,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=False,
            sep=pipelines.VID_SEG_DATASET_SEP,
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
    model_eval_dir = 'evaluation/vid_seg_predictions/%s' % config['model_name']
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name'])
    os.makedirs(model_eval_dir, exist_ok=True)
    print("generate evaluation results. ")
    model = train_utils.load(model_obj, latest_ckpt_path).to(train_utils.device)
    test_meta_path = pipelines.VID_SEG_TRAIN_SPLIT_CSV.format(
        root=config['dataset_dir'], split='test')
    df_test_meta = pd.read_csv(
        test_meta_path, dtype=str, parse_dates=False, na_values=[], keep_default_na=False)
    for _, row in tqdm(df_test_meta.iterrows(), total=len(df_test_meta), desc='test video segmentation'):
        os.makedirs(os.path.join(model_eval_dir, row[pipelines.SAMPLE_ID_KEY]), exist_ok=True)
        v = pims.Video(row[pipelines.CLIP_PATH_KEY])
        for fid in row[pipelines.ANNOTATED_FRAME_ID].split(pipelines.FRAME_ID_SEP):
            output_fname = os.path.join(model_eval_dir, row[pipelines.SAMPLE_ID_KEY]+'/'+fid+'.jpg')
            segments = model.segments([v[int(fid)]])[0]
            output_image = PilImage.fromarray(segments, mode='RGB')
            output_image.save(output_fname)


if __name__ == '__main__':
    main()
