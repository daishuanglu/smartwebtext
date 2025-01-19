import numpy as np
from tqdm import tqdm
import os
import json
import torch
import pandas as pd
import linecache
from utils.prnews import websearch
from utils import train_utils, data_utils, string_utils
from models import topic_embedding


COLS_SEP = '::::'


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


def split_data_lines(raw_data_dir, dict_path, split_files, split_fractions):
    for split_file in split_files:
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
    os.makedirs(os.path.dirname(dict_path), exist_ok=True)    
    fnames = os.listdir(raw_data_dir)
    nb_samples = [0] * len(fnames)
    word_dict = set()
    for i in tqdm(range(len(fnames)), desc='preproc data'):
        split_file = np.random.choice(split_files, p=split_fractions)
        cid = fnames[i].split('_')[-1].removesuffix('.txt')
        input_path = os.path.join(raw_data_dir, fnames[i])
        with open(split_file, 'a', encoding='utf-8') as fp:
            for texts in websearch.load_key(input_path, 'Body'):
                texts = texts.replace('\n', ' ').lower()
                normed_words = [string_utils.stemmer.stem(text) for text in texts.split()]
                word_dict = word_dict.union(normed_words)
                normed_texts = ' '.join(normed_words)
                if texts:
                    fp.write(COLS_SEP.join([normed_texts, cid]) + '\n')
                    nb_samples[i] += 1
    with open(dict_path, 'w') as fp:
        for word in sorted(normed_words):
            fp.write(word + '\n')
    return nb_samples


VAECONFIG = topic_embedding.VaeConfig(
    en1_units=128,
    en2_units=256,
    num_topic=32,
    learning_rate=0.001,
    momentum=0.99,
    init_mult=1.0,
    variance=0.995
)
COMP_URLS = 'newsdata.companyurls.txt'
KG_JSON_PATH = 'newsdata/knowledge_graph/vae_global/all.json'
CONCEPT_CSV_PATH = 'newsdata/knowledge_graph/vae_global_concepts.csv'


def main():
    config = {
        'raw_data_dir': 'newsdata/news',
        'text_col': 'text',
        'model_name': 'vae_global',
        'resume_ckpt': None,
        'train_data_path': 'training/vae_train_data.txt',
        'val_data_path': 'training/vae_val_data.txt',
        'batch_size': 64,
        'monitor': 'val_perp',
        'epochs': 10,
        'dict_path': 'training/vae_word_dict.txt',
        'skip_data_prep' : False,
        'skip_training': False
    }
    logger_dir = config.get('logger_dir', train_utils.DEFAULT_LOGGER_DIR)
    if not config.get('skip_data_prep', False):
        nb_samples = split_data_lines(
            config['raw_data_dir'],
            config['dict_path'],
            [config['train_data_path'], config['val_data_path']],
            [0.95, 0.05])
        print("create training -validation dataloader")
        train_dl = torch.utils.data.DataLoader(
            PrnewsLineDataset(config['train_data_path'], nb_samples[0]),
            collate_fn=data_utils.collate_dict,
            batch_size=config['batch_size'],
            shuffle=True)
        val_dl = torch.utils.data.DataLoader(
            PrnewsLineDataset(config['val_data_path'], nb_samples[1]),
            collate_fn=data_utils.collate_dict,
            batch_size=config['batch_size'],
            shuffle=True)
    model_obj = topic_embedding.GlobalTopicAsEmbedding(
        config, VAECONFIG).to(train_utils.device)
    if not config.get('skip_training', False):
        latest_ckpt_path = train_utils.latest_ckpt(logger_dir, config['model_name']) \
            if config['resume_ckpt'] else None
        print("model initialized. ")
        print("start training ...")
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
    eval_dl = torch.utils.data.DataLoader(
        PrnewsLineDataset(config['train_data_path'], nb_samples[0]),
        collate_fn=data_utils.collate_dict,
        batch_size=config['batch_size'],
        shuffle=False)
    os.makedirs(os.path.dirname(KG_JSON_PATH))
    devocab = {i: w for i, w in model.vocab.items()}
    cnames = [string_utils.getcompanyname(
        os.path.basename(c_url.strip())) for c_url in open(COMP_URLS, 'r')]
    pairs = []
    df = pd.DataFrame()
    for batch in tqdm(eval_dl, total=len(eval_dl), desc='extract vae concepts'):
        _, z = model(batch, compute_loss=False)
        p0 = torch.sigmoid(z)
        embeds_word_to_topic = p0.detach().cpu().numpy()
        for sample, embed in zip(batch, embeds_word_to_topic):
            eids = np.where(embed > 0.5)
            cp = cnames[sample['cid']]
            if eids.any():
                pairs.extend([{'company': cp, "action": devocab[i]} for i in eids])
                for i in eids:
                    ct = devocab[i]
                    if ct not in df.columns:
                        df[ct] = 0
                    if cp not in df.index:
                        df.at[cp, ct] = 0
                    df.at[cp, ct] += 1
    df.index.name = 'company'
    with open(KG_JSON_PATH, 'w') as fp:
        json.dump(pairs, fp)
    df.to_csv(CONCEPT_CSV_PATH)


if __name__=="__main__":
    main()