import os
import json
import pandas as pd

from models import topic_embedding
from utils import train_utils, visual_utils
from preprocessors import pipelines


OUTPUT_DIR = 'evaluation/prnews/local_topic_embedding_projection'


def load_eval_samples(eval_data_path: str, ref_col: str, query_col: str):
    df = pd.read_csv(
        eval_data_path, 
        dtype=str, 
        sep=pipelines.PRNEWS_DATA_SEP, 
        parse_dates=False, 
        keep_default_na=False, 
        na_values=[])
    df = df.sample(frac=0.01)
    return list(set(df[ref_col])), df[query_col].to_list()


def main():
    config = train_utils.read_config("experiments/news_topic/configs/prnews_local_topic_emb.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.prnews(
            output_files=[config['train_data_path'], config['val_data_path']],
            split_ratio=config['train_val_split_ratio'])
    
    logger_dir = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    model_obj = topic_embedding.LocalTopicAsEmbedding(config).to(train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name'])
    model_obj = train_utils.load(model_obj, latest_ckpt_path)
    print("model initialized. ")
    model_obj.eval()
    eval_companies, eval_news = load_eval_samples(
        config['val_data_path'], config['ref_col'], 'Title')
    ref_tok = model_obj.tokenizer(
            eval_companies,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=config['max_ref_length'])
    # Company embeddings
    company_embeddings = model_obj.cond_var_embedding(ref_tok['input_ids'].to(train_utils.device))
    company_embeddings = topic_embedding.mean_pooling(
        company_embeddings, ref_tok['attention_mask'].to(train_utils.device))
    company_embeddings = company_embeddings.detach().cpu().numpy()
    visual_utils.tensorboard_text_embedding(
        os.path.join(OUTPUT_DIR, 'companies'), eval_companies, company_embeddings)
    tok_ids_count = model_obj.doc_tok_ids_count(eval_news)
    news_embeddings = model_obj.vae.en1_fc(tok_ids_count)
    news_embeddings = news_embeddings.detach().cpu().numpy()
    visual_utils.tensorboard_text_embedding(
        os.path.join(OUTPUT_DIR, 'news'), eval_news, news_embeddings)
    

if __name__ == '__main__':
    main()