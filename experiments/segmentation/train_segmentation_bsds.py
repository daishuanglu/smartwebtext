import os
import glob
import torch
from PIL import Image as PIlImage
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from PIL import Image as PilImage
import torchvision.transforms as transforms

from preprocessors import pipelines
from utils import train_utils
from utils import data_utils
from models import segmentation


def load_bsds_image(img_path, transform_fn):
    img = PIlImage.open(img_path)
    img = transform_fn(img)
    return img


def load_bsds_label(raw_label_file, region_or_edge=0):
    # region_or_edge: 0 means region map,  1 indicates edge_map
    # BSDS uses 5 human to segment each image.
    # Each human generated their own segmentation for a same image.
    mat = sio.loadmat(raw_label_file)
    human_subject = np.random.choice(mat['groundTruth'].shape[1], 1, replace=True)[0]
    label_map = mat['groundTruth'][0, human_subject][0, 0][region_or_edge].astype('int64')
    #label_map = resize_transform(torch.from_numpy(label_map).unsqueeze(0))
    label_map = torch.from_numpy(label_map)
    return label_map.squeeze()


def main():
    config = train_utils.read_config("config/bsds_segmentation.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.bsds500(config['dataset_dir'])

    input_transform = transforms.Compose([
        transforms.Resize(config['input_size']),  # Resize to a fixed size
        transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
    ])
    train_bsds_features = {
        'image': (lambda x: load_bsds_image(str(x), input_transform)),
        'gt': (lambda x: load_bsds_label(str(x)))
    }

    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    os.makedirs(logger_dir, exist_ok=True)
    model_obj = segmentation.UNet(config).to(train_utils.device)
    print("model initialized. ")
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            pipelines.BSDS_SPLIT_CSV.format(split='train'),
            train_bsds_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        val_dl = data_utils.get_context_csv_data_loader(
            pipelines.BSDS_SPLIT_CSV.format(split='val'),
            train_bsds_features,
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
    model_eval_dir = 'evaluation/bsds_test_predictions/%s' % config['model_name']
    os.makedirs(model_eval_dir, exist_ok=True)
    print("generate evaluation results. ")
    model = train_utils.load(model_obj, latest_file)
    test_image_path = pipelines.BSDS_SAMPLES.format(root=config['dataset_dir'], split='test')
    for image_path in tqdm(glob.glob(test_image_path), desc='bsds test'):
        segments = model.segments([load_bsds_image(image_path, input_transform)])[0]
        output_fname = os.path.join(model_eval_dir, os.path.basename(image_path))
        output_image = PilImage.fromarray(segments, mode='RGB')
        output_image.save(output_fname)


if __name__ == '__main__':
    main()
