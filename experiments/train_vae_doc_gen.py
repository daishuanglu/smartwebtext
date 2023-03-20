import glob
import os
import pandas as pd

from preprocessors import pipelines
from utils import train_utils
from utils import data_utils
from models import video_text


def main():
    config = train_utils.read_config("config/vae_doc_user_gen.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.prnews(
            [config['train_data_path'], config['val_data_path']],
            config['train_val_split_ratio'], config['user_vocab_path'])

    config['logger_dir'] = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    model_obj = vae_doc_generator.ConditionalDocGenerator(config)
    print("model initialized. " )
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            config['train_data_path'],
            vae_doc_generator.train_doc_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=pipelines.PRNEWS_DATA_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None))
        val_dl = data_utils.get_context_csv_data_loader(
            config['val_data_path'], vae_doc_generator.train_doc_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=False,
            sep=pipelines.PRNEWS_DATA_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None)
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
    list_of_files = glob.glob(os.path.join(config['logger_dir'], '%s-epoch*.ckpt' % config['model_name']))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print("found checkpoint %s" % latest_file)
    print('-----------------  done training ----------------------------')
    os.makedirs('evaluation', exist_ok=True)
    print("generate evaluation results. ")
    model_obj_path = train_utils.MODEL_OBJ_PATH.format(
        logger_dir=config['logger_dir'], model_name=config['model_name'])
    model = train_utils.load(model_obj_path, latest_file)
    concepts = [
        key.split('_examples')[0] for key in config.keys() if key.endswith('_examples')]
    predictions = pd.DataFrame([], index=[n[1] for n in model.user_names])
    for c in concepts:
        eval_example_load_kwargs = {
            **config['{:s}_examples'.format(c)],
            **{'preproc_fn': pipelines.prnews_text_preproc}}
        example_sent = pipelines.eval_example_sentences(**eval_example_load_kwargs)
        print('monte carlo search test concept:', c)
        user_probs = model.monte_carlo_user(
            example_sent,
            steps=config['monte_carlo_steps'],
            batch_size=config['test_batch_size'])
        predictions[':'.join([config['model_name'],c])] = user_probs
    predictions.to_csv('evaluation/%s_eval_predictions.csv' % config['model_name'], index=False)

if __name__=="__main__":
    main()