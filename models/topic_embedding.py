import json
import pytorch_lightning as ptl
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizerFast as BertTokenizer

from utils import train_utils
import itertools


class ProdLDA(nn.Module):

    def __init__(self, net_arch, vocab_size):
        super(ProdLDA, self).__init__()
        self.net_arch = net_arch
        self.h_dim=net_arch['num_topic']
        # encoder
        self.en1_fc     = nn.Linear(vocab_size, net_arch['en1_units'])             # 1995 -> 100
        self.en2_fc     = nn.Linear(net_arch['en1_units'], net_arch['en2_units'])             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(net_arch['en2_units'], net_arch['num_topic'])             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(net_arch['num_topic'])                      # bn for mean
        self.logvar_fc  = nn.Linear(net_arch['en2_units'], net_arch['num_topic'])             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(net_arch['num_topic'])                      # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(net_arch['num_topic'], vocab_size)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(vocab_size)                      # bn for decoder

        self.a = 1 * np.ones((1, self.h_dim)).astype(np.float32)
        self.prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)  # prior_mean  = 0
        self.prior_var = torch.from_numpy(
            (((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +  # prior_var = 0.99 + 0.005 = 0.995
             (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        self.prior_logvar = self.prior_var.log()

        # initialize decoder weight
        if net_arch['init_mult'] != 0:
            self.decoder.weight.data.uniform_(0, net_arch['init_mult'])


    def forward(self, input, compute_loss=False, avg_loss=True):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        if compute_loss:
            en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn(self.mean_fc(en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p0 = F.softmax(z,dim=1)                                                # mixture probability
        p = self.p_drop(p0)
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)),dim=1)             # reconstructed distribution over vocabulary

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon, posterior_mean

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.h_dim )
        # loss
        loss = (NL + KLD)
        # in training mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss


class GlobalTopicAsEmbedding(ptl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model_name'])
        print('use bert model:', config['bert_model_name'])
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.vae = ProdLDA(config, self.vocab_size).to(train_utils.device)

    def doc_tok_ids_count(self, batch):
        tok = self.tokenizer(
            batch[self.config['text_col']],
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_doc_length'])
        tok_ids_one_hot = F.one_hot(tok['input_ids'].to(train_utils.device), self.vocab_size)
        attn_mask = torch.unsqueeze(tok['attention_mask'].to(train_utils.device), dim=-1)
        attn_mask = attn_mask.expand(tok_ids_one_hot.size()).clone()
        masked_tok_count = torch.sum(tok_ids_one_hot * attn_mask, dim=1).float()
        return masked_tok_count.to(train_utils.device)

    def embedding(self, list_of_sentences):
        self.eval()
        _, z = self({self.config['text_col']: list_of_sentences}, compute_loss=False)
        return z.detach().cpu().numpy()

    def forward(self, batch, compute_loss=False, avg_loss=True):
        tok_ids_count = self.doc_tok_ids_count(batch)
        return self.vae(tok_ids_count, compute_loss, avg_loss)

    def configure_optimizers(self):
        return [torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config['learning_rate'],
            betas=(self.config['momentum'], 0.999))], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        recon_doc, loss = self.forward(batch, compute_loss=True, avg_loss=True)
        loss_type = 'train' if training else 'val'
        log = {f'{loss_type}_perp': loss.item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss



class CondProdLDA(nn.Module):

    def __init__(self, net_arch, vocab_size, emb_size):
        super(CondProdLDA, self).__init__()
        self.net_arch = net_arch
        self.h_dim=net_arch['num_topic']
        self.cond_factor = nn.Linear(emb_size, net_arch['num_topic'])
        # encoder
        self.en1_fc     = nn.Linear(vocab_size, net_arch['en1_units'])             # 1995 -> 100
        self.en2_fc     = nn.Linear(net_arch['en1_units'], net_arch['en2_units'])             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(net_arch['en2_units'], net_arch['num_topic'])             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(net_arch['num_topic'])                      # bn for mean
        self.logvar_fc  = nn.Linear(net_arch['en2_units'], net_arch['num_topic'])             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(net_arch['num_topic'])                      # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(net_arch['num_topic'], vocab_size)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(vocab_size)                      # bn for decoder

        self.a = 1 * np.ones((1, self.h_dim)).astype(np.float32)
        self.prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)  # prior_mean  = 0
        self.prior_var = torch.from_numpy(
            (((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +  # prior_var = 0.99 + 0.005 = 0.995
             (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        self.prior_logvar = self.prior_var.log()

        # initialize decoder weight
        if net_arch['init_mult'] != 0:
            self.decoder.weight.data.uniform_(0, net_arch['init_mult'])

    def forward(self, input, cond_var_emb, compute_loss=False, avg_loss=True):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        if compute_loss:
            en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn(self.mean_fc(en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps + self.cond_factor(cond_var_emb)               # reparameterization
        p0 = F.softmax(z,dim=1)                                                # mixture probability
        p = self.p_drop(p0)
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)),dim=1)             # reconstructed distribution over vocabulary

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon, posterior_mean

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.h_dim )
        # loss
        loss = (NL + KLD)
        # in training mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LocalTopicAsEmbedding(ptl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model_name'])
        print('use bert model:', config['bert_model_name'])
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.cond_var_embedding = nn.Embedding(self.vocab_size, config['ref_emb_size'])
        self.vae = CondProdLDA(
            config, self.vocab_size, config['ref_emb_size']).to(train_utils.device)

    def doc_tok_ids_count(self, texts):
        tok = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_doc_length'])
        tok_ids_one_hot = F.one_hot(tok['input_ids'].to(train_utils.device), self.vocab_size)
        attn_mask = torch.unsqueeze(tok['attention_mask'].to(train_utils.device), dim=-1)
        attn_mask = attn_mask.expand(tok_ids_one_hot.size()).clone()
        masked_tok_count = torch.sum(tok_ids_one_hot * attn_mask, dim=1).float()
        return masked_tok_count.to(train_utils.device)

    def embedding(self, sentences, refs):
        self.eval()
        _, z = self({
            self.config['text_col']: sentences,
            self.config['ref_col']: refs
        }, compute_loss=False)
        return z.detach().cpu().numpy()

    def forward(self, batch, compute_loss=False, avg_loss=True):
        ref_tok = self.tokenizer(
            batch[self.config['ref_col']],
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_ref_length'])
        ref_emb = self.cond_var_embedding(ref_tok['input_ids'])
        ref_emb = mean_pooling(ref_emb, ref_tok['attention_mask'])
        tok_ids_count = self.doc_tok_ids_count(batch[self.config['text_col']])
        return self.vae(tok_ids_count, ref_emb, compute_loss, avg_loss)

    def configure_optimizers(self):
        return [torch.optim.AdamW(
            params=self.parameters(),
            lr=self.config['learning_rate'],
            betas=(self.config['momentum'], 0.999))], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        recon_doc, loss = self.forward(batch, compute_loss=True, avg_loss=True)
        loss_type = 'train' if training else 'val'
        log = {f'{loss_type}_perp': loss.item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss