import os
import pandas as pd

from preprocessors import pipelines
from utils import train_utils, data_utils
from models import cf


def main():
    config = train_utils.read_config("config/prnews_tte_sent_small.yaml")

    train_sentence_features = {
        config['ref_col']: (lambda x: str(x)),
        config['query_col']: (lambda x: str(x)),
        config['score_col']: (lambda x: 1.0 if float(x) > 0 else 0.0)}

    if not config.get("skip_prep_data", False):
        pipelines.prnews(
            output_files=[config['train_data_path'], config['val_data_path']],
            split_ratio=config['train_val_split_ratio'],
            vocabs={config['item_vocab_path']: [config['ref_col']]})
        pipelines.prnews_negatives(
            data_path=config['train_data_path'],
            ref_col=config['ref_col'],
            score_col=config['score_col'],
            neg_ratio=config['neg_sample_ratio'])
        pipelines.prnews_negatives(
            data_path=config['val_data_path'],
            ref_col=config['ref_col'],
            score_col=config['score_col'],
            neg_ratio=config['neg_sample_ratio'])

    logger_dir = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    model_obj = cf.TTEModel(config).to(train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
        if config['resume_ckpt'] else None
    print("model initialized. ")
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            config['train_data_path'],
            train_sentence_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=pipelines.PRNEWS_DATA_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None))
        val_dl = data_utils.get_context_csv_data_loader(
            config['val_data_path'],
            train_sentence_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=False,
            sep=pipelines.PRNEWS_DATA_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
        print("start training ...")
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
    os.makedirs(pipelines.PRNEWS_EVAL_DIR, exist_ok=True)
    print("generate evaluation results. ")
    model = train_utils.load(model_obj, latest_ckpt_path)
    df_val = pd.read_csv(
        config['val_data_path'],
        sep=pipelines.PRNEWS_DATA_SEP,
        dtype=str, parse_dates=False, na_values=[], keep_default_na=False)
    companies = df_val[config['ref_col']].unique().tolist()
    test_keywords = ['analytics', 'innovation', 'technology']
    predictions = pd.DataFrame(
        data=[],
        columns=[config['model_name'] + ':' + kw for kw in test_keywords],
        index=companies)
    for kw in test_keywords:
        user_embed, item_embed = model(
            {config['query_col']: [kw]* len(companies), config['ref_col']: companies})
        scores = (user_embed * item_embed).sum(dim=1).detach().cpu().numpy()
        predictions[config['model_name'] + ':' + kw] = (scores + 1)/2 # normalize cosine score to [0,1]
    predictions.to_csv(os.path.join(
        pipelines.PRNEWS_EVAL_DIR, '%s_val_predictions.csv' % config['model_name']))

if __name__=="__main__":
    main()