import pytorch_lightning as ptl
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from collections import defaultdict
import numpy as np
from data_utils import get_data_loader
from abc import ABC
from train_utils import training_pipeline
from train_utils import save as cf_save


def load(model, PATH):
    #model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

def save(model, PATH):
    torch.save(model.state_dict(), PATH)
    return


train_base_features = {
    'user_id': (lambda x: x.values.astype(np.float32)),
    'item_id': (lambda x: x.values.astype(np.float32)),
    'score': (lambda x: x.values.astype(np.float32))
}

eval_base_features={
    'user_id': (lambda x: x.values.astype(np.float32)),
    'item_id': (lambda x: x.values.astype(np.float32))
}

class TTEModel(ptl.LightningModule, ABC):
    def __init__(self,config, n_user_bins, n_store_bins):
        super(TTEModel, self).__init__(config)
        self.user_id_column = "user_id"
        self.store_id_column = "item_id"
        self.embedding_user = nn.Embedding(
            n_user_bins,
            self.config['embed_dim']
        )
        self.embedding_store = nn.Embedding(
            n_store_bins,
            self.config['embed_dim']
        )
        self.n_user_bins = n_user_bins
        self.n_store_bins = n_store_bins
        self.metrics_k_all = self.config['metrics_k_all']
        self.learning_rate = self.config['learning_rate']
        self.mse = nn.MSELoss(reduction='mean')

    def query_model(self, batch):
        idx = batch[self.user_id_column].long() % self.n_user_bins
        e = self.embedding_user(idx).squeeze(1)
        return F.normalize(e, p=2.0, dim=1)

    def item_model(self, batch):
        idx = batch[self.store_id_column].long() % self.n_store_bins
        e = self.embedding_store(idx).squeeze(1)
        return F.normalize(e, p=2.0, dim=1)

    def forward(self, batch):
        return self.query_model(batch), self.item_model(batch)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.learning_rate)], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        user_embed, item_embed = self.forward(batch)
        predictions = (user_embed * item_embed).sum(dim=1)
        loss = self.mse(predictions, batch[self.score_column])
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, on_step=True, prog_bar=True)
        return loss


def tte_predict_pipeline(model, test_x, source_mat=None):
    df = defaultdict(list)
    start_time = time.time()
    for ib, batch in enumerate(iter(test_x)):
        user_embed, item_embed = model(batch)
        score = (user_embed * item_embed).sum(dim=1).detach().numpy()
        col = batch['user_id'].cpu().detach().numpy().astype(int)
        row = batch['item_id'].cpu().detach().numpy().astype(int)
        if source_mat is not None:
            #ind=np.vstack((np.atleast_2d(row), np.atleast_2d(col))).transpose()
            source_mat[row,col] = score
        else:
            df['user_id'] += col.tolist()
            df['item_id'] += row.tolist()
            df['score'] += score.tolist()
        print('prediction batch %d, %d secs.' % (ib, int(time.time()-start_time)))
    if source_mat is not None: return source_mat
    else: return df


def impute(imputor_config, cf, uni_emb, emb_mask):
    test_loader = get_data_loader(
        uni_emb,
        1 - emb_mask,
        features=eval_base_features,
        name="test",
        batch_size=imputor_config['batch_size']
    )
    uni_emb_imp = np.memmap(
        imputor_config['imputed_emb_path'],
        dtype='float32',
        mode='w+',
        shape=uni_emb.shape)
    uni_emb_imp[:] = tte_predict_pipeline(cf, test_loader, uni_emb)[:]
    uni_emb_imp.flush()
    return np.memmap(
            imputor_config['imputed_emb_path'],
            dtype='float32',
            mode='c',
            shape=uni_emb.shape
    )


def imputor(imputor_config, uni_emb, emb_mask, tmp_data_path):
    n_users, n_items = uni_emb.shape
    cf = TTEModel(
        imputor_config,
        n_user_bins=n_users,
        n_store_bins=n_items
    )
    if not imputor_config['load_pretrained']:
        train_loader = get_data_loader(
            uni_emb,
            emb_mask,
            features=train_base_features,
            name="train",
            batch_size=imputor_config['batch_size'],
            temp_data_path=tmp_data_path
        )
        val_loader = get_data_loader(
            uni_emb[:100, :100],
            emb_mask[:100, :100],
            features=train_base_features,
            name="val",
            batch_size=imputor_config['batch_size'],
            temp_data_path=tmp_data_path
        )
        cf = training_pipeline(
            cf,
            train_loader,
            val_loader,
            imputor_config['epochs'],
        )
        cf_save(cf, imputor_config['model_path'])
    else:
        cf = load(cf, imputor_config['model_path'])
    return cf