import os

from preprocessors import pipelines
from utils import train_utils
from utils import data_utils
from models import video_text


def main():
    # TODO: Hanyi see if you can train a detr video model as follows
    config = train_utils.read_config("config/detr_kth_action_vid.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.kth_action_video()

    config['logger_dir'] = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    model_obj = video_text.Detr(config)
    print("model initialized. ")
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            config['train_data_path'],
            video_text.train_kth_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=pipelines.KTH_ACTION_DATA_CSV_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None),
            col_fns=video_text.kth_detr_col_fns
        )
        val_dl = data_utils.get_context_csv_data_loader(
            config['val_data_path'], video_text.train_kth_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=False,
            sep=pipelines.KTH_ACTION_DATA_CSV_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None),
            col_fns=video_text.kth_detr_col_fns
        )
        logger_dir = config.get("logger_dir", "./lightning_logs")
        os.makedirs(logger_dir, exist_ok=True)
        print("start training ...")
        model_obj, _ = train_utils.training_pipeline(
            model_obj,
            train_dl,
            val_x=val_dl,
            nepochs=config['epochs'],
            resume_ckpt=config['resume_ckpt'],
            model_name=config['model_name'],
            monitor=config['monitor'],
            logger_path=config['logger_dir'])
