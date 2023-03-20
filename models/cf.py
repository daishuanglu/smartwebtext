import gc

import numpy as np
import json
import pandas as pd
import torch
import pytorch_lightning as ptl
from abc import ABC
import torch.nn.functional as F
from typing import Dict
from collections import defaultdict
from transformers import DistilBertTokenizer as BertTokenizer
from transformers import DistilBertModel as BertModel
#from transformers import MobileBertTokenizer as BertTokenizer
#from transformers import MobileBertModel as BertModel
from sentence_transformers import SentenceTransformer, models, losses
from torch import nn
from sentence_transformers import evaluation
from sentence_transformers import util as sbert_util
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from itertools import product
from train_utils import  read_config, device

#bert_model_name = 'google/mobilebert-uncased'
bert_model_name = 'distilbert-base-uncased'
print('Use BERT model: ', bert_model_name)

train_sentence_features = {
    'Company': (lambda x: str(x)),
    'Text': (lambda x: str(x)),
    #'scores': (lambda x: 1.0-float(x) if float(x)>0 else 0.0) # higlight less frequent words in the news.
    'scores': (lambda x: 1.0 if float(x)>0 else 0.0)
}

train_word_features = {
    'company': (lambda x: str(x)),
    'phrase': (lambda x: str(x)),
    'scores': (lambda x: 1.0-float(x) if float(x)>0 else 0.0) # higlight less frequent words in the news.
    #'scores': (lambda x: 1.0 if float(x)>0 else 0.0)
}

train_phrase_features = {
    'Company': (lambda x: str(x)),
    'phrase': (lambda x: str(x)),
    "Text": (lambda x: str(x)),
    'scores': (lambda x: 1.0-float(x) if float(x)>0 else 0.0) # higlight less frequent words in the news.
}

train_sbert_features = {
    'Company': (lambda x: (str(x), 0)),
    'phrase': (lambda x: (str(x), 1)),
    "Text": (lambda x: (str(x), 0)),
    'scores': (lambda x: (1.0-float(x) if float(x)>0 else 0.0, 2)) # higlight less frequent words in the news.
}

eval_phrase_features={
    'Company': (lambda x: str(x)),
    'phrase': (lambda x: str(x)),
    "Text": (lambda x: str(x))
}

"""
Context info. based Two tower embedding model
"""
def max_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def flatten_gt(gt_path, ref_col, query_col, score_col):
    df_gt = pd.read_csv(gt_path)
    df = defaultdict(list)
    words_col = [w for w in df_gt.columns if w not in ['company', 'tic']]
    for _, row in df_gt.iterrows():
        for w in words_col:
            df[ref_col].append(row['company'])
            df[query_col].append(w)
            df[score_col].append(row[w])
    df = pd.DataFrame.from_dict(df)
    print(df)
    return df


class SemiSupTTEModel(ptl.LightningModule, ABC):
    def __init__(self,config):
        super().__init__()
        if isinstance(config, str):
            self.config = read_config(config)
        else:
            self.config = config
        with open(self.config['item_vocab_path'], 'r') as f:
            self.item_vocab = json.load(f)
        self.mse = nn.MSELoss(reduction='mean')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.query_transformer = BertModel.from_pretrained(
            bert_model_name, return_dict=True).to(device)
        self.item_embedding = nn.Embedding(len(self.item_vocab), self.config['embed_size'])
        self.semi_gt = flatten_gt(
            self.config['semi_gt_path'],
            self.config['ref_col'],
            self.config['query_col'],
            self.config['score_col'])
        self.gt_querys = sorted( set(self.semi_gt[self.config['query_col']]))
        #self.gt_refs = sorted(set(self.semi_gt[self.config['ref_col']]))
        self.semi_gt.set_index([self.config['ref_col'], self.config['query_col']], inplace=True)
        #self._query_gt = []

    def query_model(self, batch):
        #self._query_gt = [ list(set(q.split()).intersection(self.gt_querys))\
        #                           for q in batch[self.config['query_col']]]
        batch_text = batch[self.config['query_col']]
        encoding = self.tokenizer(
            batch_text ,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_query_length']
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        e = self.query_transformer(input_ids, attention_mask=attention_mask)
        e = mean_pooling(e[0], attention_mask)[:, :self.config['embed_size']]
        return F.normalize(e, p=2.0, dim=1)

    def item_model(self, batch):
        batch_items = [self.item_vocab[w] for w in batch[self.config['ref_col']]]
        batch_items = torch.tensor(batch_items).to(device)
        e = self.item_embedding(batch_items)
        return F.normalize(e, p=2.0, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.query_model(batch), self.item_model(batch)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        query_embed, item_embed = self.forward(batch)
        items = batch[self.config['ref_col']]
        gt_query_embed = self.query_model({self.config['query_col']: self.gt_querys})
        cos_sent_to_gt_query = sbert_util.cos_sim(query_embed, gt_query_embed)
        cos_sent_to_gt_query, ind = cos_sent_to_gt_query.max(dim=1)
        cos_sent_to_gt_query = cos_sent_to_gt_query.detach()
        label_scores = [self.semi_gt.loc[(items[i], self.gt_querys[iq])][self.config['score_col']]
                        for i, iq in enumerate(ind.detach().cpu().numpy())]
        # negative cosine related: prediction --> -cos_sent_to_gt_query
        # positive cosine related: prediction --> +cos_sent_to_gt_query
        label_scores = torch.tensor(label_scores).float().to(device) *2 -1
        cos_sent_to_item = (query_embed * item_embed).sum(dim=1)
        #  add label penalty term.
        loss = self.mse((cos_sent_to_item+1)/2, batch[self.config['score_col']]) +\
               10.0 * self.mse(cos_sent_to_item, label_scores*cos_sent_to_gt_query)
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss


class TTEBertItem(ptl.LightningModule, ABC):
    def __init__(self,config):
        super().__init__()
        if isinstance(config, str):
            self.config = read_config(config)
        else:
            self.config = config
        self.mse = nn.MSELoss(reduction='mean')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.query_transformer = BertModel.from_pretrained(
            bert_model_name, return_dict=True).to(device)
        self.item_transformer = BertModel.from_pretrained(
            bert_model_name, return_dict=True).to(device)

    def query_model(self, batch):
        batch_text = batch[self.config['query_col']]
        encoding = self.tokenizer(
            batch_text ,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_query_length']
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        e = self.query_transformer(input_ids, attention_mask=attention_mask)
        e = mean_pooling(e[0], attention_mask)[:, :self.config['embed_size']]
        return F.normalize(e, p=2.0, dim=1)

    def item_model(self, batch):
        batch_text = batch[self.config['ref_col']]
        encoding = self.tokenizer(
            batch_text,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_query_length']
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        e = self.item_transformer(input_ids, attention_mask=attention_mask)
        e = mean_pooling(e[0], attention_mask)[:, :self.config['embed_size']]
        return F.normalize(e, p=2.0, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.query_model(batch), self.item_model(batch)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        user_embed, item_embed = self.forward(batch)
        predictions = (user_embed * item_embed).sum(dim=1)
        loss = self.mse(predictions, batch[self.config["score_col"]])
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss


class TTEModel(ptl.LightningModule, ABC):
    def __init__(self,config):
        super().__init__()
        if isinstance(config, str):
            self.config = read_config(config)
        else:
            self.config = config
        with open(self.config['item_vocab_path'], 'r') as f:
            self.item_vocab = json.load(f)
        self.mse = nn.MSELoss(reduction='mean')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.query_transformer = BertModel.from_pretrained(
            bert_model_name, return_dict=True).to(device)
        self.item_embedding = nn.Embedding(len(self.item_vocab), self.config['embed_size'])

    def query_model(self, batch):
        batch_text = batch[self.config['query_col']]
        encoding = self.tokenizer(
            batch_text ,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_query_length']
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        e = self.query_transformer(input_ids, attention_mask=attention_mask)
        e = mean_pooling(e[0], attention_mask)[:, :self.config['embed_size']]
        return F.normalize(e, p=2.0, dim=1)

    def item_model(self, batch):
        batch_items = [self.item_vocab[w] for w in batch[self.config['ref_col']]]
        batch_items = torch.tensor(batch_items).to(device)
        e = self.item_embedding(batch_items)
        return F.normalize(e, p=2.0, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.query_model(batch), self.item_model(batch)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        user_embed, item_embed = self.forward(batch)
        predictions = (user_embed * item_embed).sum(dim=1)
        loss = self.mse(predictions, batch[self.config["score_col"]])
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss


class TTEContextModel(ptl.LightningModule, ABC):
    def __init__(self,config):
        super().__init__()
        if isinstance(config, str):
            self.config = read_config(config)
        else:
            self.config = config
        self.mse = nn.MSELoss(reduction='mean')
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.query_transformer = BertModel.from_pretrained(
            bert_model_name, return_dict=True).to(device)
        self.context_transformer = BertModel.from_pretrained(
            bert_model_name, return_dict=True).to(device)
        if self.config['sentence_pooling']=="mean_pooling":
            self.sent_pooling_i = mean_pooling
            self.sent_pooling_q = mean_pooling
        elif self.config['sentence_pooling']=="max_pooling":
            self.sent_pooling_i = max_pooling
            self.sent_pooling_q = max_pooling
        else:
            self.sent_pooling_i = mean_pooling
            self.sent_pooling_q = mean_pooling
        #embed_size = self.query_transformer.embeddings.word_embeddings.weight.shape[1]
        #self.sent_batch_norm = nn.BatchNorm1d(embed_size)

    def query_model(self, batch):
        batch_text = batch[self.config['query_col']]
        encoding = self.tokenizer(
            batch_text ,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_sequence_length']
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        e = self.query_transformer(input_ids, attention_mask=attention_mask)
        sent_e = self.sent_pooling_q(e[0], attention_mask)
        return F.normalize(sent_e, p=2.0, dim=1)
        #return sent_e

    def item_model(self, batch):
        #batch_context =['[SEP]'.join(w) for w in zip(*[batch[col] for col in self.context_cols])]
        context_tok = self.tokenizer(
            batch[self.config['context_col']],
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_sequence_length']
        )
        context_tok_ids = context_tok['input_ids'].to(device)
        context_tok_attention_mask = context_tok['attention_mask'].to(device)
        context_e = self.context_transformer(
            context_tok_ids, attention_mask=context_tok_attention_mask)
        ref_tok = self.tokenizer(
            batch[self.config['ref_col']],
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_sequence_length']
        )
        ref_tok_ids = ref_tok['input_ids'].to(device)
        ref_tok_attention_mask = ref_tok['attention_mask'].to(device)
        ref_e = self.context_transformer(
            ref_tok_ids, attention_mask=ref_tok_attention_mask)
        context_sent_e = self.sent_pooling_i(context_e[0], context_tok_attention_mask)
        ref_sent_e = self.sent_pooling_i(ref_e[0], ref_tok_attention_mask)
        return F.normalize(context_sent_e+ ref_sent_e, p=2.0, dim=1)
        #return context_sent_e + ref_sent_e

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.query_model(batch), self.item_model(batch)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        user_embed, item_embed = self.forward(batch)
        predictions = (user_embed * item_embed).sum(dim=1)
        loss = self.mse(predictions, batch[self.config["score_col"]])
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss


class OTEContextModel(ptl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, str):
            self.config = read_config(config)
        else:
            self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.transformer_encoder = BertModel.from_pretrained(
            bert_model_name, return_dict=True).to(device)
        if self.config['sentence_pooling']=="mean_pooling":
            self.sent_pooling = mean_pooling
        elif self.config['sentence_pooling']=="max_pooling":
            self.sent_pooling = max_pooling
        else:
            self.sent_pooling = mean_pooling
        embed_size = self.transformer_encoder.embeddings.word_embeddings.weight.shape[1]
        self.linear = nn.Linear(embed_size, 1)

    def forward(self, batch):
        batch_context = ['[SEP]'.join(w) for w in zip(*[batch[col]\
            for col in [self.config['ref_col'], self.config['context_col'], self.config['query_col'] ]
                                                        ])]
        encoding = self.tokenizer(
            batch_context,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_sequence_length']
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        e = self.transformer_encoder(input_ids, attention_mask=attention_mask)
        sent_e = self.sent_pooling(e[0], attention_mask)
        self_sim = self.linear(sent_e)
        return self_sim

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.learning_rate)], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        predictions = self.forward(batch)
        loss = self.mse(predictions, batch[self.config['score_col']])
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, on_step=True, prog_bar=True)
        return loss

def _load_ref_dataset(kws, query_col, ref_col, context_col, df_eval, eval_company, limit=None):
    df_eval = df_eval[df_eval[ref_col].isin(eval_company)]
    print("MTurk eval on %d companies." % len(set(df_eval[ref_col])))
    print(df_eval[ref_col].value_counts())
    eval_batch = {
        query_col: [],
        ref_col: [],
        context_col: []}
    df_ref_context = df_eval[[ref_col, context_col]].drop_duplicates()
    df_ref_context = df_ref_context.dropna()
    if limit is not None:
        df_ref_context = df_ref_context[:limit]
    eval_batch[query_col] = kws
    eval_batch[ref_col] = df_ref_context[
        ref_col].apply(lambda x: str(x)).to_list()
    eval_batch[context_col] = df_ref_context[context_col].apply(lambda x: str(x)).to_list()
    return eval_batch


class SBertContextModel():
    def __init__(self, config):
        self.config = read_config(config) if isinstance(config, str) else config
        model = SentenceTransformer('paraphrase-albert-small-v2')
        word_embedding_model = model._first_module()
        tokens = [self.config['special_sep']]
        word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.config['embed_size'],
            activation_function=nn.Tanh())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=device)
        self.train_loss = losses.CosineSimilarityLoss(self.model)

    def train(self, train_x, val_ds):
        evaluator = evaluation.EmbeddingSimilarityEvaluator(val_ds[0], val_ds[1], val_ds[2])
        self.model.fit(
            train_objectives=[(train_x, self.train_loss)],
            epochs=self.config['epochs'],
            evaluator=evaluator,
            evaluation_steps=1000,
            warmup_steps=self.config['warmup_steps'],
            output_path=self.config['saved_model_path'],
            save_best_model=True)


def vec2mat_sim(vec,mat):
    vec=np.atleast_2d(vec)
    mat=mat.transpose()
    p1 = vec.dot(mat)
    return p1


def _chunk_predict(model, eval_batch, ref_col, query_col, n_items, chunk =256):
    i=0
    ref_embed= []
    query_embeds = []
    for i in tqdm(range(0, n_items, chunk)):
        query_embed, item_embed = model(
            {ref_col: eval_batch[ref_col][i:chunk+i],
            query_col: eval_batch[query_col][i:chunk+i]})
        item_embed = item_embed.detach().cpu().numpy()
        query_embed = query_embed.detach().cpu().numpy()
        ref_embed.append(item_embed)
        query_embeds.append(query_embed)
        i += chunk
        #print('predict %d/%d items, %d secs' % (i, n_items, int(time.time()-start_time)))
    if i<n_items:
        query_embed, item_embed = model(
            {ref_col: eval_batch[ref_col][i:chunk + i],
            query_col: eval_batch[query_col][i:chunk + i]})
        item_embed = item_embed.detach().cpu().numpy()
        query_embed = query_embed.detach().cpu().numpy()
        ref_embed.append(item_embed)
        query_embeds.append(query_embed)
    ref_embed = np.concatenate(ref_embed, axis=0)
    query_embeds = np.concatenate(query_embeds, axis=0)
    sim = np.sum(query_embeds * ref_embed, axis=1)
    return sim


def get_unflatten_fn(ref_col, query_col, score_col, method_name='tte'):
    method_name= method_name if method_name else 'unknown_method'

    def _pred_col(query_term):
        col_name = ':'.join([method_name, query_term])
        return col_name

    from string_utils import find_links
    from ticker_utils import ticker_finder
    def _unflatten_predictions(df):
        kws = sorted(set(df[query_col]))
        df_unflatten = defaultdict(list)
        items = sorted(set(df[ref_col]))
        df_unflatten[ref_col] = items
        for q in kws:
            df_unflatten[_pred_col(q)]= [0.0 for _ in range(len(items))]
        #df_unflatten[score_col] = [0.0 for _ in range(len(items))]
        for _, row in df.iterrows():
            id = df_unflatten[ref_col].index(row[ref_col])
            df_unflatten[_pred_col(row[query_col])][id] = row[score_col]
        df_unflatten = pd.DataFrame.from_dict(df_unflatten)
        df_unflatten['link'] = find_links(df_unflatten[ref_col], 'ais/default_search_full.txt')
        df_unflatten['tic'] = df_unflatten[ref_col].apply(ticker_finder)
        return df_unflatten
    return _unflatten_predictions

"""
Input a list of kws, a prediction function always return a dataframe with [score_col, query_col, ref_col]
"""
def get_tte_search_prediction_fn(
        train_data_path, val_data_path, model, ref_col, query_col, score_col, unflatten_fn=None):
    dfs = []
    if train_data_path is not None:
        dfs.append(pd.read_csv(train_data_path))
    if val_data_path is not None:
        dfs.append(pd.read_csv(val_data_path))
    ref_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    refs = sorted(set(ref_df[ref_col]))
    nsamples = len(ref_df)

    def _prediction_fn(kws, unflatten=False, hasword_only=True):
        pred_dfs = []
        for iw, word in enumerate(kws):
            hasword = ref_df[query_col].apply(lambda x: word in str(x))
            print('+ search for %s, %d/%d' % (word, iw, len(kws)))
            print('%d/%d query "%s" has "%s".' % (hasword.sum(), nsamples, query_col, word))
            if hasword_only:
                ref_df_local = ref_df[hasword].copy()
                n_items = hasword.sum()
            else:
                ref_df_local = ref_df.copy()
                n_items = len(hasword)
            eval_batch = {
                query_col: ref_df_local[query_col].tolist(),
                ref_col: ref_df_local[ref_col].tolist()}
            print('run model inference')
            word_scores = _chunk_predict(model, eval_batch, ref_col, query_col, n_items, chunk=1024)
            pred_df = pd.DataFrame.from_dict(
                {ref_col: ref_df_local[ref_col].tolist(),
                 score_col: (word_scores+1)/2,
                 query_col: [word for _ in range(n_items)]}
            )
            max_idx = pred_df.groupby(ref_col)[score_col].transform(max) == pred_df[score_col]
            pred_df = pred_df[max_idx]
            # if the query word not exists in the training-validation dataset
            # (with neither negative nor positive labels) query column. Then we force the score to be 0s.
            null_ref = [r for r in refs if r not in set(pred_df[ref_col])]
            if null_ref:
                null_df = pd.DataFrame.from_dict({ref_col: null_ref,
                     score_col: [0.0 for _ in range(len(null_ref))],
                     query_col: [word for _ in range(len(null_ref))]})
                pred_df = pd.concat([pred_df, null_df], ignore_index=True)
            pred_df = pred_df.sort_values(by=[ref_col])
            pred_dfs.append(pred_df)
        pred_dfs = pd.concat(pred_dfs, ignore_index=True)
        unflatten = False if unflatten_fn is None else unflatten
        if unflatten:
            return unflatten_fn(pred_dfs)
        else:
            return pred_dfs

    return _prediction_fn

def get_tte_prediction_fn(refs, model, ref_col, query_col, score_col, unflatten_fn=None):
    def _prediction_fn(kws, unflatten=False):
        print('%d flattened query ' % len(kws))
        ref_df = defaultdict(list)
        for q, r in product(*[kws, refs]):
            ref_df[query_col].append(q)
            ref_df[ref_col].append(r)
            ref_df[score_col].append(0.0)
        ref_df = pd.DataFrame(ref_df)
        n_items = len(ref_df)
        eval_batch = {
            query_col: ref_df[query_col].tolist(),
            ref_col: ref_df[ref_col].tolist()}
        print('run model inference')
        word_scores = _chunk_predict(model, eval_batch, ref_col, query_col, n_items, chunk=1024)
        pred_df = pd.DataFrame.from_dict(
            {ref_col: ref_df[ref_col].tolist(),
            score_col: (word_scores+1)/2,
            query_col: ref_df[query_col].tolist()}
        )
        pred_df = pred_df.sort_values(by=[ref_col])
        unflatten = False if unflatten_fn is None else unflatten
        if unflatten:
            return unflatten_fn(pred_df)
        else:
            return pred_df

    return _prediction_fn
