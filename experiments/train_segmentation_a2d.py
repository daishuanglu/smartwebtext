import os
import glob
import torch

import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image as PilImage
import torchvision.transforms as transforms
import pims

from preprocessors import pipelines
from utils import train_utils
from utils import data_utils
from models import segmentation


def load_a2d_frame_images(dataset_dir, feature_dict, transform_fn):
    fids = [int(fid) for fid in feature_dict['fids'].split(pipelines.A2D_FID_SEP)]
    vf = pipelines.A2D_CLIP_PATH.format(root=dataset_dir, vid=feature_dict['vid'])
    v = pims.Video(vf)
    imgs = [transform_fn(PilImage.fromarray(v[i])) for i in fids]
    return imgs


def load_a2d_label_map(dataset_dir, feature_dict):
    fids = [fid for fid in feature_dict['fids'].split(pipelines.A2D_FID_SEP)]
    label_maps = []
    for fid in fids:
        fmat = pipelines.A2D_ANNOTATION_PATH.format(
            root=dataset_dir, vid=feature_dict['vid'], fid=fid)
        with h5py.File(fmat, 'r') as mat_file:
            mat = np.array(mat_file['reS_id'])
        label_maps.append(torch.from_numpy(mat).long())
    return label_maps


def main():
    config = train_utils.read_config("config/a2d_video_segmentation.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.a2d_video_images(
            config['dataset_dir'], config['label_colors_json'], config['train_val_ratio'])

    input_transform = transforms.Compose([
        transforms.Resize(config['input_size']),  # Resize to a fixed size
        transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
    ])
    train_vid_features = {
        'fids': (lambda x: str(x)),
        'vid': (lambda x: str(x))
    }
    train_a2d_cols = {
        'image': lambda x: load_a2d_frame_images(config['dataset_dir'], x, input_transform),
        'gt': lambda x: load_a2d_label_map(config['dataset_dir'], x)}

    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    os.makedirs(logger_dir, exist_ok=True)
    model_obj = segmentation.UNet(config).to(train_utils.device)
    print("model initialized. ")
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
            sep=pipelines.KTH_ACTION_DATA_CSV_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        print("start training, logging at %s" % logger_dir)
        model_obj, _ = train_utils.training_pipeline(
            model_obj,
            train_dl,
            val_x=val_dl,
            nepochs=config['epochs'],
            resume_ckpt=config['resume_ckpt'],
            model_name=config['model_name'],
            monitor=config['monitor'],
            logger_path=logger_dir)

    list_of_files = glob.glob(os.path.join(config['logger_dir'], '%s-epoch*.ckpt' % config[
        'model_name']))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print("found checkpoint %s" % latest_file)
    print('-----------------  done training ----------------------------')
    model_eval_dir = 'evaluation/A2D_test_predictions/%s' % config['model_name']
    os.makedirs(model_eval_dir, exist_ok=True)
    print("generate evaluation results. ")
    model = train_utils.load(model_obj, latest_file)
    df_test_meta = pipelines.A2D_IMAGE_SPLIT_CSV.format(root=config['dataset_dir'], split='test')
    for _, row in tqdm(df_test_meta.iterrows(), desc='A2D test video segmentation'):
        os.makedirs(os.path.join(model_eval_dir, row['VID']), exist_ok=True)
        vf = pipelines.A2D_CLIP_PATH.format(vid=row['VID'])
        v = pims.Video(vf)
        for fid in row['fids'].split(pipelines.A2D_FID_SEP):
            output_fname = os.path.join(model_eval_dir, row['VID']+'/'+fid+'.jpg')
            segments = model.segments([input_transform(v[int(fid)])])[0]
            output_image = PilImage.fromarray(segments, mode='RGB')
            output_image.save(output_fname)


if __name__ == '__main__':
    main()
