import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from preprocessors import pipelines
from utils import train_utils, data_utils, metric_utils
from models import topic_embedding


def main():
    config = train_utils.read_config("config/prnews_global_topic_emb.yaml")

    train_doc_features = {
        config['ref_col']: (lambda x: str(x)),
        config['text_col']: (lambda x: str(x))}

    if not config.get("skip_prep_data", False):
        pipelines.prnews(
            output_files=[config['train_data_path'], config['val_data_path']],
            split_ratio=config['train_val_split_ratio'])

    logger_dir = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    model_obj = topic_embedding.GlobalTopicAsEmbedding(config).to(train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
        if config['resume_ckpt'] else None
    print("model initialized. ")
    if not config.get('skip_training', False):
        print("create training -validation dataloader")
        train_dl = data_utils.get_context_csv_data_loader(
            config['train_data_path'],
            train_doc_features,
            batch_size=config['batch_size'],
            clear_cache=config['clear_cache'],
            shuffle=True,
            sep=pipelines.PRNEWS_DATA_SEP,
            max_line=10 ** 7,
            limit=config.get('limit', None))
        val_dl = data_utils.get_context_csv_data_loader(
            config['val_data_path'],
            train_doc_features,
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
    kw_embeddings = {}
    for kw in test_keywords:
        kw_embeddings[kw] = model.embedding([kw])
    predictions = pd.DataFrame(
        data=[],
        columns=[config['model_name']+':'+kw for kw in test_keywords],
        index=companies)
    predictions = predictions.fillna(0.0)
    for c in tqdm(companies,
                  desc='validation prediction %d companies' % len(companies)):
        df_val_c = df_val[df_val[config['ref_col']]==c]
        for _, row in df_val_c.iterrows():
            prnews_emb = model.embedding([row[config['text_col']]])
            for kw in test_keywords:
                kw_sim = metric_utils.pw_cos_sim(kw_embeddings[kw], prnews_emb)[0,0]
                kw_sim = (kw_sim + 1) / 2
                if kw_sim > predictions.at[c, config['model_name']+':'+kw]:
                    predictions.at[c, config['model_name']+':'+kw] = kw_sim
    print(predictions)
    predictions.index.name = 'company'
    predictions.to_csv(os.path.join(
        pipelines.PRNEWS_EVAL_DIR, '%s_val_predictions.csv' % config['model_name']))

if __name__=="__main__":
    main()