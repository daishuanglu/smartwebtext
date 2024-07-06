import os
import json

from models import cf
from utils import train_utils, visual_utils
from preprocessors import pipelines
from evaluators.prnews import assets


OUTPUT_DIR = 'evaluation/prnews/tte_embedding_projection'


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
    visual_utils.tensorboard_text_embedding(
        os.path.join(OUTPUT_DIR, 'companies'), all_companies, company_embeddings)
    news_embeddings = model_obj.query_model({
        config['ref_col']: assets.TEST_NEWS_QUERIES})
    visual_utils.tensorboard_text_embedding(
        os.path.join(OUTPUT_DIR, 'news'), assets.TEST_NEWS_QUERIES, news_embeddings)
    