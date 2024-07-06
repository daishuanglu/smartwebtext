import json

from models import topic_embedding
from utils import train_utils
from preprocessors import pipelines


TTE_ITEM_VOCAB_PATH = ''


def main():
    config = train_utils.read_config("config/prnews_local_topic_emb.yaml")
    if not config.get("skip_prep_data", False):
        pipelines.prnews(
            output_files=[config['train_data_path'], config['val_data_path']],
            split_ratio=config['train_val_split_ratio'])
    
    with open(TTE_ITEM_VOCAB_PATH, 'r') as f:
        ref_vocab = json.load(f)
    logger_dir = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    model_obj = topic_embedding.LocalTopicAsEmbedding(config).to(train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name'])
    model_obj = train_utils.load(model_obj, latest_ckpt_path)
    print("model initialized. ")
    model_obj.eval()
    ref_tok = model_obj.tokenizer(
            list(ref_vocab),
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=config['max_ref_length'])
    ref_emb = model_obj.cond_var_embedding(ref_tok['input_ids'].to(train_utils.device))
    ref_emb = topic_embedding.mean_pooling(
        ref_emb, ref_tok['attention_mask'].to(train_utils.device))
    # Company embeddings
    all_companies = ['' for _ in ref_vocab]
    for name, idx in ref_vocab.items():
        all_companies[idx] = name
    company_embeddings = ref_emb
    # Press news query embeddings.
    news_embeddings = model_obj.embedding(
        sentences=['sentence1', 'sentence2', 'sentence3'],
        refs=['company1'])
    
    