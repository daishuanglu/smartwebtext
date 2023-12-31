import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image as PilImage
import pims
from preprocessors import pipelines
from utils import train_utils, data_utils
from models import segmentation
from preprocessors.video_annotation import VideoMetadata


def load_frame_images(feature_dict):
    fids = [int(fid) for fid in VideoMetadata.frame_ids(feature_dict)]
    vf =feature_dict[pipelines.CLIP_PATH_KEY]
    v = pims.Video(vf)
    samples = [torch.from_numpy(v[i]) for i in fids]
    #samples = [PilImage.fromarray(v[i]) for i in fids]
    return samples


def load_frame_annotation(feature_dict, unique_cls_id_map={}):
    obj_masks = []
    label_masks = []
    ref_texts = []
    annotations = VideoMetadata.frame_annotations(
        feature_dict, unique_cls_id_map[VideoMetadata.dataset_name(feature_dict)])
    for context in annotations:
        obj_masks.append(torch.from_numpy(context.obj_mask).long())
        label_masks.append(torch.from_numpy(context.label_mask).long())
        ref_texts.append(context.ref_text)
    return {'obj_mask': obj_masks, 'ref_text': ref_texts, 'label_mask': label_masks}


def main():
    config = train_utils.read_config(
        "experiments/segmentation/configs/pix2pix_video_segmentation.yaml")
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
        pipelines.DATASET_KEY: (lambda x: str(x))
    }
    unique_cls_id_map = VideoMetadata.unique_class_id_map(split='')
    train_a2d_cols = {
        'frames': lambda x: load_frame_images(x),
        'gt_frames': lambda x: load_frame_annotation(x, unique_cls_id_map)
    }
    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    os.makedirs(logger_dir, exist_ok=True)
    model_obj = segmentation.BaseSegmentor(
        config,
        multi_fname_sep=pipelines.FRAME_ID_SEP).to(train_utils.device)
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
    test_meta_path = pipelines.VID_SEG_TRAIN_SPLIT_CSV.format(split='test')
    df_test_meta = pd.read_csv(
        test_meta_path, sep=pipelines.VID_SEG_DATASET_SEP, dtype=str, parse_dates=False,
        na_values=[], keep_default_na=False)
    n_eval_visuals = config.get('n_eval_visuals', 10)
    i_eval = 0
    for _, row in tqdm(df_test_meta.iterrows(), total=len(df_test_meta), desc='eval video segmentation'):
        v = pims.Video(row[pipelines.CLIP_PATH_KEY])
        frames = []
        for fid in row[pipelines.ANNOTATED_FRAME_ID].split(pipelines.FRAME_ID_SEP):
            frames.append(torch.from_numpy(v[int(fid)]).to(train_utils.device))
        vid = VideoMetadata.video_id(row)
        fids = VideoMetadata.frame_ids(row)
        batch = {'frames': [frames]}
        pred_dict = model(batch)
        pred_dict = {k: v[0] for k, v in pred_dict.items() if v}
        output_fname = os.path.join(model_eval_dir, vid + '.h5')
        VideoMetadata.save_sequence_predictions(
            pred_dict, fids, output_fname, model.bg_conf_thresh, model.conf_thresh)
        if i_eval < n_eval_visuals:
            os.makedirs(os.path.join(model_eval_dir, vid), exist_ok=True)
            for i, fid in enumerate(fids):
                output_cont_path = os.path.join(
                    model_eval_dir, vid + '/' + fid + '_cont.jpg')
                im_conts = model.to_contours(pred_dict['prob'][i], v[int(fid)])
                PilImage.fromarray(im_conts, mode='RGB').save(output_cont_path)
                output_seg_path = os.path.join(
                    model_eval_dir, vid + '/' + fid + '_seg.jpg')
                im_color_seg, _ = model.to_label_colors(pred_dict['prob'][i], pred_dict['cls_prob'][i])
                PilImage.fromarray(im_color_seg.astype('uint8')).save(output_seg_path)
        i_eval += 1

if __name__ == '__main__':
    main()
