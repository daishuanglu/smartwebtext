import numpy as np
from tqdm import tqdm
import os
import re
import json
import torch
import pandas as pd
import linecache
import collections
from typing import List
from utils.prnews import websearch
from utils import train_utils, data_utils, string_utils
from models import topic_embedding


COLS_SEP = '::::'
STOPWORDS = string_utils.load_stopwords()


def load_concept_df_from_json_pairs(pair_json_files: List[str], cnames = None):
    df = pd.DataFrame()
    for json_file in pair_json_files:
        cid = int(os.path.splitext(os.path.basename(json_file))[0])
        with open(json_file, 'r') as fp:
            pairs = json.load(fp)
        for p in pairs:
            ct, cp = p['action'], p['company']
            if cnames is not None:
                cp = cnames[cid]
            if ct not in df.columns:
                df[ct] = 0
            if cp not in df.index:
                df.at[cp, ct] = 0
            df.at[cp, ct] += 1
    df.index.name = 'company'
    df = df.fillna(0)
    return df


class PrnewsLineDataset(torch.utils.data.Dataset):
    def __init__(self, path, nb_samples, clrCache=True, max_line = 10**7):
        self.fpath = path
        self.len = nb_samples
        self.nb_samples = nb_samples
        self.clrCache = clrCache
        self.cnt = 0
        self.max_line = max_line

    def __getitem__(self, index):
        irow = index + 1
        line = linecache.getline(self.fpath, irow)
        line = line.strip().split(COLS_SEP)
        if self.cnt == self.max_line and self.clrCache:
            linecache.clearcache()
            self.cnt = 0
        try:
            self.cnt += 1
            return {'text': line[0], 'cid': line[1]}
        except Exception as e:
            print(e)
            print(index,  irow, self.fpath)
            assert False, e.__repr__()

    def __len__(self):
        return self.len


def split_data_lines(raw_data_dir, split_files, split_fractions, dict_path=None, vocab_size=4096,
                     limit=None):
    for split_file in split_files:
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
    mode = 'a'
    if dict_path is not None:
        os.makedirs(os.path.dirname(dict_path), exist_ok=True) 
        mode = 'w'
    fnames = os.listdir(raw_data_dir)
    nb_samples = [0] * len(fnames)
    word_dict = collections.Counter()
    for i in tqdm(range(len(fnames)), desc='preproc data'):
        if limit is not None:
            if i > limit:
                break
        split_file = np.random.choice(split_files, p=split_fractions)
        cid = fnames[i].split('_')[-1].removesuffix('.txt')
        input_path = os.path.join(raw_data_dir, fnames[i])
        with open(split_file, mode, encoding='utf-8') as fp:
            for texts in websearch.load_key(input_path, 'Body'):
                # preprocessing
                lines = texts.lower().strip().split('\n')
                lines = [re.sub('[^A-Za-z ]', ' ', line) for line in lines]
                lines = [string_utils.remove_stopwords(line, STOPWORDS).strip() for line in lines]
                for line in lines:
                    if line:
                        normed_words = [string_utils.stemmer.stem(word) for word in line.split()]
                        word_dict.update(normed_words)
                        line = ' '.join(normed_words)
                        fp.write(COLS_SEP.join([line, cid]) + '\n')
                        nb_samples[i] += 1
    if dict_path is not None:
        with open(dict_path, 'w') as fp:
            for word, cnt in word_dict.most_common(vocab_size): 
                fp.write(word + '\n')
    return nb_samples


VAECONFIG = topic_embedding.VaeConfig(
    en1_units=128,
    en2_units=256,
    num_topic=32,
    init_mult=1.0,
    variance=0.995,
    vocab_size=4096
)
COMP_URLS = 'newsdata/companyurls.txt'
KG_JSON_DIR = 'newsdata/knowledge_graph/vae_global'
CONCEPT_CSV_PATH = 'newsdata/knowledge_graph/vae_global_concept.csv'
LIMIT = 34


def main():
    config = {
        'raw_data_dir': 'newsdata/news',
        'text_col': 'text',
        'model_name': 'vae_global',
        'resume_ckpt': None,
        'train_data_path': 'training/vae_train_data.txt',
        'val_data_path': 'training/vae_val_data.txt',
        'eval_data_path': 'training/vae_eval_data.txt',
        'batch_size': 2048,
        'monitor': 'val_perp',
        'epochs': 10,
        'dict_path': 'training/vae_word_dict.txt',
        'skip_data_prep' : True,
        'skip_training': True,
        'learning_rate' : 0.0008,
        'momentum': [0.9, 0.999],
    }
    logger_dir = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    if not config.get('skip_data_prep', False):
        split_data_lines(
            raw_data_dir=config['raw_data_dir'],
            split_files=[config['train_data_path'], config['val_data_path']],
            split_fractions=[0.95, 0.05],
            vocab_size=VAECONFIG.vocab_size,
            dict_path=config['dict_path'])
    model_obj = topic_embedding.GlobalTopicAsEmbedding(
        config, VAECONFIG).to(train_utils.device)
    latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
            if config['resume_ckpt'] else None
    if not config.get('skip_training', False):
        nb_train_samples = sum(1 for _ in open(config['train_data_path'], 'r'))
        train_dl = torch.utils.data.DataLoader(
            PrnewsLineDataset(config['train_data_path'], nb_train_samples),
            collate_fn=data_utils.collate_dict,
            batch_size=config['batch_size'],
            shuffle=True)
        nb_val_samples = sum(1 for _ in open(config['val_data_path'], 'r'))
        val_dl = torch.utils.data.DataLoader(
            PrnewsLineDataset(config['val_data_path'], nb_val_samples),
            collate_fn=data_utils.collate_dict,
            batch_size=config['batch_size'],
            shuffle=True)
        model, _ = train_utils.training_pipeline(
            model_obj,
            train_dl,
            val_x=val_dl,
            nepochs=config['epochs'],
            resume_ckpt=latest_ckpt_path,
            model_name=config['model_name'],
            monitor=config['monitor'],
            logger_path=logger_dir)
    else:
        latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name'])
        model = train_utils.load(model_obj, latest_ckpt_path)
    model.eval()
    split_data_lines(
        raw_data_dir=config['raw_data_dir'],
        split_files=[config['eval_data_path']],
        split_fractions=[1.0],
        dict_path=None,
        vocab_size=VAECONFIG.vocab_size,
        limit=LIMIT)
    nb_eval_samples = sum(1 for _ in open(config['eval_data_path'], 'r'))
    eval_dl = torch.utils.data.DataLoader(
        PrnewsLineDataset(config['eval_data_path'], nb_eval_samples),
        collate_fn=data_utils.collate_dict,
        batch_size=config['batch_size'],
        shuffle=False)
    os.makedirs(KG_JSON_DIR, exist_ok=True)
    vocab_words = ['[unk]' for _ in range(model.vocab_size)]
    devocab = {i: w for w, i in model.vocab.items()}
    for i, w in devocab.items():
        vocab_words[i] = w
    embed_vocab_words = model.embedding(vocab_words).cpu().detach()
    embed_vocab_words = torch.nn.functional.normalize(embed_vocab_words, p=2, dim=1)
    cnames = [string_utils.getcompanyname(
        os.path.basename(c_url.strip('/\n'))) for c_url in open(COMP_URLS, 'r')]
    ucids = set()
    res_sim = {i: torch.tensor([0] * model.vocab_size) for i in range(len(cnames))}
    for batch in tqdm(eval_dl, total=len(eval_dl), desc='extract vae concepts'):
        embed_docs = model.embedding(batch[config['text_col']]).cpu().detach()
        embed_docs = torch.nn.functional.normalize(embed_docs, p=2, dim=1)
        cos_sim = torch.matmul(embed_docs, embed_vocab_words.transpose(1, 0))
        sim_scores = (cos_sim + 1) / 2
        for i, cid in enumerate(batch['cid']):
            cid = int(cid)
            #print(sim_scores[i, :])
            res_sim[cid] = torch.maximum(sim_scores[i, :], res_sim[cid])
            ucids.add(cid)
    
    pairs = {}
    np = 0
    json_pairs = set()
    for cid in ucids:
        related_cpt = [vocab_words[i] for i in range(model.vocab_size - 1) 
                       if 0.9 > res_sim[cid][i].item() > 0.85 and (len(vocab_words[i]) > 3)]
        json_pairs.add(f'{KG_JSON_DIR}/{cid}.json')
        if os.path.exists(f'{KG_JSON_DIR}/{cid}.json'):
            continue
        if cid not in pairs:
            pairs[cid] = []
        print("company[", cnames[cid], "], related_cpt", related_cpt)
        cpairs = [{'company': cnames[cid], "action": ct} for ct in related_cpt]
        pairs[cid].extend(cpairs)
        np += len(related_cpt)
    
    print(np, ' total concept pairs.')
    for cid in ucids:
        with open(f'{KG_JSON_DIR}/{cid}.json', 'w') as fp:
            json.dump(pairs[cid], fp)
    df = load_concept_df_from_json_pairs(json_pairs, cnames)
    df.to_csv(CONCEPT_CSV_PATH)


if __name__=="__main__":
    main()
