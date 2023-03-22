import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as ptl
from abc import ABC
import json
from tqdm import tqdm
from transformers import BertTokenizerFast as BertTokenizer
from utils.train_utils import device


train_doc_features = {
    'Company': (lambda x: str(x)),
    'Text': (lambda x: str(x))}


class ProdLDA(nn.Module):

    def __init__(self, config, vocab_size, cond_vocab_size):
        super(ProdLDA, self).__init__()
        self.num_cond_var_vals = cond_vocab_size
        self.h_dim=config['num_topic']
        # encoder
        self.en1_fc     = nn.Linear(vocab_size, config['en1_units'])             # 1995 -> 100
        self.en2_fc     = nn.Linear(config['en1_units'], config['en2_units'])             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(config['en2_units'], config['num_topic']*2)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(config['num_topic']*2)                      # bn for mean
        self.logvar_fc  = nn.Linear(config['en2_units'], config['num_topic']*2)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(config['num_topic']*2)                      # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        self.cond_p_drop = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(config['num_topic'], vocab_size)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(vocab_size)                      # bn for decoder
        # conditional decoder
        self.cond_decoder = nn.Linear(config['num_topic'], cond_vocab_size)  # 50   -> 3374 companies
        self.cond_decoder_bn = nn.BatchNorm1d(cond_vocab_size)
        self.a = 1 * np.ones((1, self.h_dim*2)).astype(np.float32)
        self.prior_mean = torch.from_numpy((np.log(self.a).T - np.mean(np.log(self.a), 1)).T)  # prior_mean  = 0
        self.prior_var = torch.from_numpy(
            (((1.0 / self.a) * (1 - (2.0 / self.h_dim))).T +  # prior_var = 0.99 + 0.005 = 0.995
             (1.0 / (self.h_dim * self.h_dim)) * np.sum(1.0 / self.a, 1)).T)
        self.prior_logvar = self.prior_var.log()

        # initialize decoder weight
        if config['init_mult'] != 0:
            self.decoder.weight.data.uniform_(0, config['init_mult'])


    def forward(self, input, cond_var=None, compute_loss=False, avg_loss=True):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn(self.mean_fc(en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_())  # noise
        z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
        z, z_cond = z[:, :self.h_dim], z[:, self.h_dim:]
        p0 = F.softmax(z, dim=1)  # mixture probability
        p = self.p_drop(p0)
        p_cond = F.softmax(z_cond, dim=1)  # mixture probability
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)), dim=1)
        recon_cond = F.softmax(self.cond_decoder_bn(self.cond_decoder(p_cond)), dim=1)
        if compute_loss:
            return recon, recon_cond, self.loss(
                input, recon, recon_cond, cond_var, posterior_mean, posterior_logvar, posterior_var, avg_loss)
        else:
            return recon, recon_cond, (p, p_cond)

    def loss(
            self, input, recon, recon_cond, cond_var, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017,
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean).to(device)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean).to(device)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean).to(device)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.h_dim )
        # loss
        loss = (NL + KLD)
        if cond_var is not None:
            cond_var = F.one_hot(cond_var, self.num_cond_var_vals)
            NL_user = -(cond_var * (recon_cond + 1e-10).log()).sum(1)
            loss += NL_user
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss


class ConditionalDocGenerator(ptl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_model_name'])
        print('use bert model:', config['bert_model_name'])
        self.vocab_size = len(self.tokenizer.get_vocab())
        with open(config['user_vocab_path'], 'r') as f:
            self.user_vocab = json.load(f)
            self.user_names = sorted([(id, w) for w, id in self.user_vocab.items()])
        self.vae = ProdLDA(config, self.vocab_size, len(self.user_vocab)).to(device)

    def doc_tok_ids_count(self, batch):
        tok = self.tokenizer(
            batch[self.config['text_col']],
            return_tensors='pt',
            padding=True, truncation=True,
            max_length=self.config['max_doc_length'])
        tok_ids_one_hot = F.one_hot(tok['input_ids'].to(device), self.vocab_size)
        attn_mask = torch.unsqueeze(tok['attention_mask'].to(device), dim=-1)
        attn_mask = attn_mask.expand(tok_ids_one_hot.size()).clone()
        masked_tok_count = torch.sum(tok_ids_one_hot * attn_mask, dim=1).float()
        return masked_tok_count.to(device)

    def generate_user(self, list_of_doc_strs):
        batch = {self.config['text_col']: list_of_doc_strs}
        recon_doc, recon_user, probs = self(batch, False, False)
        return recon_user.cpu().detach().numpy()

    def monte_carlo_user(self, sentences, steps=1000, batch_size=1024):
        self.eval()
        user_probs = np.zeros((len(sentences), len(self.user_vocab)))
        for _ in tqdm(range(steps)):
            for i in range(0, len(sentences), batch_size):
                u_particles = self.generate_user(sentences[i: i+batch_size])
                user_probs[i: i+batch_size, :] += u_particles
        user_probs /= steps
        user_probs = user_probs.max(axis=0)
        return user_probs

    def forward(self, batch, compute_loss=False, avg_loss=True):
        tok_ids_count = self.doc_tok_ids_count(batch)
        user_ids = None
        if compute_loss:
            user_ids = torch.tensor(
                [self.user_vocab[r] for r in batch[self.config['ref_col']]]).to(device)
        return self.vae(tok_ids_count, user_ids, compute_loss, avg_loss)

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
        recon_doc, recon_user, loss = self.forward(batch, compute_loss=True, avg_loss=True)
        loss_type = 'train' if training else 'val'
        log = {f'{loss_type}_perp': loss.item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss



