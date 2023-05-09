import os

from preprocessors import pipelines
from utils import train_utils
from utils import data_utils
from models import segmentation


def main():
    config = train_utils.read_config("config/bsds_segmentation.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.bsds500(config['dataset_dir'])

    logger_dir = config.get("logger_dir", train_utils.DEFAULT_LOGGER_DIR)
    model_obj = segmentation.UNet(config)
    print("model initialized. ")
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            pipelines.BSDS_SPLIT_CSV.format(split='train'),
            segmentation.train_bsds_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        val_dl = data_utils.get_context_csv_data_loader(
            pipelines.BSDS_SPLIT_CSV.format(split='val'),
            segmentation.train_bsds_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=False,
            sep=pipelines.KTH_ACTION_DATA_CSV_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        os.makedirs(logger_dir, exist_ok=True)
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

if __name__ == '__main__':
    main()
