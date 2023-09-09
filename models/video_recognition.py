import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as ptl
from abc import ABC
from typing import Dict
from transformers import VivitImageProcessor, VivitModel
import torch.nn as nn
import torch.nn.functional as F

from utils import video_utils

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    nominator = torch.sum(token_embeddings * input_mask_expanded, 1)
    denominator = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return nominator / denominator


def heats_cube(heats, cubelet_size, cube_size):
    heats = heats[:, 1:]
    bsize, n_cubelets = heats.size()
    # 16 * 14 * 14 -> 16 * 224 * 224
    nrows = cube_size[0]//cubelet_size
    ncols = cube_size[1]//cubelet_size
    nlen = cube_size[2]//cubelet_size
    heats = heats.view(bsize, nrows, ncols, nlen)
    #heats = heats.transpose(3, 1)
    heats = heats.unsqueeze(1)
    # bsize* 1 * 14 * 14 * 16 -> bsize * 1 * 224*224*256
    heats = F.upsample(heats, scale_factor=cubelet_size, mode='nearest')
    heats = heats.squeeze(1)
    #heats = heats.transpose(1, 3)
    return heats


class Vivit(ptl.LightningModule, ABC):
    def __init__(self, config, video_key, target_key, num_classes):
        super().__init__()
        self.video_key = video_key
        self.target_key = target_key
        self.num_classes = num_classes
        self.config = config
        self.image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.dense_pool_emb = nn.Linear(768, num_classes)
        self.nll = nn.NLLLoss()
        self.ce = nn.CrossEntropyLoss()
        self.df_predictions = []
        self.epoch = 0
        self.val_cam = None
        os.makedirs(os.path.dirname(self.config['val_prediction_fstr']), exist_ok=True)

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        inputs = self.image_processor(batch[self.video_key], return_tensors="pt")
        outputs = self.model(**inputs)
        # batch_size * 3137 (16*16*16 + 1) * embed_size (768)
        encoded_seq = outputs.last_hidden_state
        p_encoded_seq = torch.sigmoid(encoded_seq)
        #cls_emb = p_encoded_seq[:, 0, :]
        #pooled_seq = torch.mean(p_encoded_seq[:, 1:], dim=1) + cls_emb
        pooled_seq = torch.mean(p_encoded_seq, dim=1)
        logits = self.dense_pool_emb(pooled_seq)
        p = F.softmax(logits, dim=-1)
        cam = self.dense_pool_emb(encoded_seq)
        loss = None
        if compute_loss:
            targets = torch.tensor(batch[self.target_key]).to(p.device)
            loss = self.ce(logits, targets.long())
        return {'p': p, 'cam': cam, 'loss': loss}

    def training_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        loss_fn = 'crossent_map' if self.config.get('map_loss_factor', 0.0) > 0 else 'crossent'
        log = {f'{loss_fn}_loss': outputs['loss'].item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        predictions = outputs['p'].argmax(dim=-1)
        df_pred = pd.DataFrame({'target': batch[self.target_key],
                                'predictions': predictions.cpu().detach().numpy(),
                                'vid_path': batch['vid_path'],
                                'className': batch['className']})
        df_pred['is_correct'] = df_pred['target'] == df_pred['predictions']
        self.df_predictions.append(df_pred)
        log = {'batch_error_rate': 1 - df_pred['is_correct'].mean()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        if self.global_step % self.config.get('val_clip_steps', 1000) == 0:
            val_video_fpath = self.config['val_prediction_fvid'].format(iter=self.global_step,
                                                                        vid='{:s}')
            os.makedirs(os.path.dirname(val_video_fpath), exist_ok=True)
            batch_cams = outputs['cam'][torch.arange(outputs['cam'].size(0)), :, predictions]
            heatmaps = heats_cube(batch_cams, cube_size=(224, 224, 256), cubelet_size=16)
            for hm, clip, vid in zip(heatmaps, batch[self.video_key], batch['id']):
                blended_heatmap = video_utils.video_alpha_blending(
                    hm.detach().cpu().numpy(), clip, frame_size=(224, 224))
                video_utils.save3d(val_video_fpath.format(vid), blended_heatmap)
        return log['batch_error_rate']

    def on_validation_epoch_end(self):
        df_pred = pd.concat(self.df_predictions)
        epoch_error_rate = 1 - df_pred['is_correct'].mean()
        log = {'val_error_rate': epoch_error_rate}
        self.log_dict(log, batch_size=self.config['batch_size'], prog_bar=True)
        output_path = self.config['val_prediction_fstr'].format(epoch=self.epoch)
        df_pred.to_csv(output_path, index=False)
        self.df_predictions = []
        self.epoch +=1

        return log['val_error_rate']

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []
