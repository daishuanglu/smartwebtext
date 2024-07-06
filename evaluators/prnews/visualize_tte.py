import json
import random

from models import cf
from utils import train_utils
from preprocessors import pipelines

import umap


if __name__ == '__main__':
    config = train_utils.read_config("experiments/news_topic/configs/prnews_tte_sent_small.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.prnews(
            output_files=[config['train_data_path'], config['val_data_path']],
            split_ratio=config['train_val_split_ratio'],
            vocabs={config['item_vocab_path']: [config['ref_col']]})
    with open(config['item_vocab_path'], 'r') as f:
        ref_vocab = json.load(f)
    ref_vocab = list(ref_vocab.keys())
    logger_dir = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    model_obj = cf.TTEModel(config).to(train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name'])
    model_obj = train_utils.load(model_obj, latest_ckpt_path)
    model_obj.eval()
    # Company embeddings
    all_companies = ['' for _ in ref_vocab]
    for name, idx in ref_vocab.items():
        all_companies[idx] = name
    company_embeddings = model_obj.item_embedding.data.weights
    # Press news query embeddings.
    news_embeddings = model_obj.query_model({
        config['ref_col']: [
            'query1',
            'query2'
            ]})

    
    embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(digits.data)


    