
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as ptl
import pandas as pd
from transformers import DistilBertTokenizer as BertTokenizer
from transformers import DistilBertModel as BertModel
import data_utils, train_utils
import yaml
from constants import LOGGER_DIR
from tqdm import tqdm


train_features = {
    'id': lambda x: str(x),
    'buildingblock1_smiles': lambda x: str(x),
    'buildingblock2_smiles': lambda x: str(x),
    'buildingblock3_smiles': lambda x: str(x),
    'molecule_smiles': lambda x: str(x),
    'protein_name': lambda x: str(x),
    'binds': lambda x: float(x)
}

FEATURES = ['buildingblock1_smiles',
            'buildingblock2_smiles',
            'buildingblock3_smiles',
            'molecule_smiles',
            'protein_name']

train_cols = {
    'text': lambda x: '[SEP]'.join([x[feat] for feat in FEATURES])
    }
OUTPUT_PATH = 'belka_test_predictions.csv'
BERT_MODEL = 'distilbert-base-uncased'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BelkaModel(ptl.LightningModule):

    def __init__(self, config) -> None:
        super(BelkaModel, self).__init__()
        self.config = config
        self.loss_fn = nn.MSELoss()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.text_transformer = BertModel.from_pretrained(
            BERT_MODEL, return_dict=True).to(DEVICE)
        self.max_input_length = config['max_input_length']
        self.embed_size = config['embed_size']
        self.linear = nn.Linear(config['embed_size'], 1).to(DEVICE)

    def text_encoder_input(self, list_of_strings):
        encoding = self.tokenizer(list_of_strings,
                                  return_tensors="pt",
                                  padding="max_length",
                                  truncation=True,
                                  max_length=self.max_input_length)
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
        return encoding

    def forward(self, batch):
        inputs = self.text_encoder_input(batch['text'])
        outputs = self.text_transformer(**inputs)
        enc = mean_pooling(outputs['last_hidden_state'], inputs['attention_mask'])
        score = self.linear(enc)
        return torch.sigmoid(score).squeeze(-1)

    def configure_optimizers(self):
        return [torch.optim.AdamW(
            params=self.parameters(), lr=self.config['learning_rate'])], []

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def train_or_val_step(self, batch, training):
        scores = self.forward(batch)
        loss_type = 'train' if training else 'val'
        target = torch.tensor(batch['binds']).to(DEVICE).clone().detach()
        #weights = (target == 0) * 0.5 + (target == 1) * 99.5
        #loss = torch.mean(weights * ((scores - target) ** 2))
        loss = self.loss_fn(scores, target)
        log = {f'{loss_type}_loss': loss.item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss


def read_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    config = read_config('configs/belka.yaml')
    #cnt = {'1': 0, '0': 0}
    #with open(config['train_data_path'], 'r') as f:
    #    next(f)
    #    for line in tqdm(f, total=300000000):
    #        cnt[line[-2]] += 1
    # binds count in training: {'1': 1589906, '0': 293656924}
    #print('binds count in training:', cnt)
    train_dl = data_utils.get_context_csv_data_loader(
            config['train_data_path'],
            train_features,
            batch_size=config['batch_size'],
            clear_cache=True,
            shuffle=True,
            sep=',',
            col_fns=train_cols,
            max_line=10 ** 7,
            limit=config.get('limit', None))
    val_dl = data_utils.get_context_csv_data_loader(
            config['val_data_path'],
            train_features,
            batch_size=config['batch_size'],
            clear_cache=True,
            shuffle=False,
            sep=',',
            col_fns=train_cols,
            max_line=10 ** 7,
            limit=config.get('limit', None))
    model_obj = BelkaModel(config).to(DEVICE)
    print("start training ...")
    model_obj, _ = train_utils.training_pipeline(
            model_obj,
            train_dl,
            val_x=val_dl,
            nepochs=config['epochs'],
            model_name=config['model_name'],
            resume_ckpt=config.get('resume_ckpt', ''),
            monitor=config['monitor'],
            logger_path=LOGGER_DIR)
    test_dl = data_utils.get_context_csv_data_loader(
            config['test_data_path'],
            train_features,
            batch_size=config['batch_size'],
            clear_cache=True,
            shuffle=False,
            col_fns=train_cols,
            sep=',',
            max_line=10 ** 7,
            limit=config.get('limit', None)
        )
    latest_ckpt_path = train_utils.latest_ckpt(LOGGER_DIR, config['model_name'])
    model = train_utils.load(model_obj, latest_ckpt_path).to(DEVICE)
    model.eval()
    nsamples = sum(1 for _ in open(config['test_data_path'], 'r'))
    nlimit = config.get('limit', nsamples)
    nsamples = min(nlimit, nsamples)
    results = {'id': [], 'binds': []}
    for batch in tqdm(test_dl, desc='eval prediction', total=nsamples):
        outputs = model(batch)
        results['id'] += batch['id']
        scores = model(batch)
        scores = scores.detach().cpu().numpy().tolist()
        results['binds'] += scores
    pd.DataFrame(results).set_index('id').to_csv(OUTPUT_PATH)