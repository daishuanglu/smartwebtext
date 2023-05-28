import os
import glob

import pandas as pd
import torch

import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image as PilImage
import pims

from preprocessors import pipelines
from utils import train_utils
from utils import data_utils
from utils import image_utils
from models import segmentation


def pytorch_to_tensor(images):
    tensor_images = [torch.from_numpy(image.transpose((2, 0, 1))).float() for image in images]
    tensor_images = torch.stack(tensor_images)
    tensor_images /= 255.0
    return tensor_images


def load_a2d_frame_image_patches(dataset_dir, feature_dict, patch_size=(256,256)):
    fids = [int(fid) for fid in feature_dict['fids'].split(pipelines.A2D_FID_SEP)]
    vf = pipelines.A2D_CLIP_PATH.format(root=dataset_dir, vid=feature_dict['vid'])
    v = pims.Video(vf)
    imgs = []
    patch_coords = []
    for i in fids:
        np_imgs, patch_coordinates = image_utils.image_to_patches(v[i], tuple(patch_size))
        imgs.append([torch.from_numpy(np_img) for np_img in np_imgs])
        patch_coords.append(patch_coordinates)
    return imgs, patch_coords


def load_a2d_label_map_patches(dataset_dir, feature_dict, patch_size=(256, 256)):
    fids = [fid for fid in feature_dict['fids'].split(pipelines.A2D_FID_SEP)]
    label_maps = []
    patch_coords = []
    for fid in fids:
        fmat = pipelines.A2D_ANNOTATION_PATH.format(
            root=dataset_dir, vid=feature_dict['vid'], fid=fid)
        with h5py.File(fmat, 'r') as mat_file:
            mat = np.array(mat_file['reS_id']).T
            np_maps, patch_coord = image_utils.image_to_patches(mat, tuple(patch_size))
            label_map = [torch.from_numpy(np_map).long() for np_map in np_maps]
            patch_coords.append(patch_coord)
        label_maps.append(label_map)
    return label_maps, patch_coords


def main():
    config = train_utils.read_config("config/a2d_video_segmentation.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.a2d_video_images(
            config['dataset_dir'], config['label_colors_json'], config['train_val_ratio'])

    train_vid_features = {
        'fids': (lambda x: str(x)),
        'vid': (lambda x: str(x))
    }
    train_a2d_cols = {
        'image': lambda x: load_a2d_frame_image_patches(config['dataset_dir'], x)[0],
        'gt': lambda x: load_a2d_label_map_patches(config['dataset_dir'], x)[0],
        'patch_coords': lambda x: load_a2d_label_map_patches(config['dataset_dir'], x)[1]
    }

    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    os.makedirs(logger_dir, exist_ok=True)
    model_obj = segmentation.Pix2Pix(
        config, multi_fname_sep=pipelines.A2D_FID_SEP).to(train_utils.device)
    print("model initialized. ")
    list_of_files = glob.glob(os.path.join(logger_dir, '%s-epoch*.ckpt' % config[
        'model_name']))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print("found checkpoint %s" % latest_file)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
        if config['resume_ckpt'] else None
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            pipelines.A2D_IMAGE_SPLIT_CSV.format(split='train'),
            train_vid_features,
            col_fns=train_a2d_cols,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        val_dl = data_utils.get_context_csv_data_loader(
            pipelines.A2D_IMAGE_SPLIT_CSV.format(split='val'),
            train_vid_features,
            col_fns=train_a2d_cols,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
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
    model_eval_dir = 'evaluation/A2D_test_predictions/%s' % config['model_name']
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name'])
    os.makedirs(model_eval_dir, exist_ok=True)
    print("generate evaluation results. ")
    model = train_utils.load(model_obj, latest_ckpt_path)
    test_meta_path = pipelines.A2D_IMAGE_SPLIT_CSV.format(root=config['dataset_dir'], split='test')
    df_test_meta = pd.read_csv(
        test_meta_path, dtype=str, parse_dates=False, na_values=[], keep_default_na=False)
    for _, row in tqdm(df_test_meta.iterrows(), desc='A2D test video segmentation'):
        os.makedirs(os.path.join(model_eval_dir, row['vid']), exist_ok=True)
        vf = pipelines.A2D_CLIP_PATH.format(root=config['dataset_dir'], vid=row['vid'])
        v = pims.Video(vf)
        for fid in row['fids'].split(pipelines.A2D_FID_SEP):
            output_fname = os.path.join(model_eval_dir, row['vid']+'/'+fid+'.jpg')
            segments = model.segments([v[int(fid)]])[0]
            output_image = PilImage.fromarray(segments, mode='RGB')
            output_image.save(output_fname)


if __name__ == '__main__':
    main()
