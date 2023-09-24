import os
from preprocessors import pipelines
import pandas as pd
import torch
import pytorch_lightning as ptl
from abc import ABC
from typing import Dict
import pims
import numpy as np
from transformers import VivitImageProcessor, VivitModel
import torch.nn as nn
import torch.nn.functional as F

from utils import video_utils, train_utils

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    nominator = torch.sum(token_embeddings * input_mask_expanded, 1)
    denominator = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return nominator / denominator


def heats_cube(heats, cubelet_size, cube_size, target_size=None):
    heats = heats[:, 1:]
    bsize, n_cubelets = heats.size()
    # 16 * 14 * 14 -> 16 * 224 * 224
    nrows = cube_size[0]//cubelet_size[0]
    ncols = cube_size[1]//cubelet_size[1]
    nlen = cube_size[2]//cubelet_size[2]
    heats = heats.view(bsize, nlen, nrows, ncols)
    #heats = heats.transpose(3, 1)
    heats = heats.unsqueeze(1)
    if target_size is None:
        target_size = cube_size
    # bsize* 1 * 14 * 14 * 16 -> bsize * 1 * 224*224*256
    # mini-batch x channels x [optional depth] x [optional height] x width.
    heats = F.interpolate(heats, size=target_size[::-1], mode='trilinear', align_corners=False)
    heats = heats.transpose(2, 4)
    return heats.squeeze(1)


class Vivit(ptl.LightningModule, ABC):
    def __init__(self, config, video_key, target_key, num_classes):
        super().__init__()
        self.video_key = video_key
        self.target_key = target_key
        self.num_classes = num_classes
        self.config = config
        self.image_processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.dense_pool_emb = nn.Linear(768, num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.df_predictions = []
        self.epoch = 0
        self.val_cam = None
        self.val_steps = 0
        self.val_pred_dir = os.path.dirname(self.config['val_prediction_fstr'])
        os.makedirs(self.val_pred_dir, exist_ok=True)

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        inputs = self.image_processor(batch[self.video_key], return_tensors="pt")
        inputs = inputs.to(train_utils.device)
        outputs = self.model(**inputs)
        # batch_size * 3137 (16*16*16 + 1) * embed_size (768)
        encoded_seq = outputs.last_hidden_state
        p_encoded_seq = torch.sigmoid(encoded_seq)
        # batch_size * embed_size (768)
        pooled_seq = torch.mean(p_encoded_seq, dim=1)
        logits = self.dense_pool_emb(pooled_seq)
        p = F.softmax(logits, dim=-1)
        cam = self.dense_pool_emb(p_encoded_seq)
        loss = None
        if compute_loss:
            targets = torch.tensor(batch[self.target_key]).to(p.device)
            loss = self.ce(logits, targets.long())
        return {'p': p.cpu().detach(),
                'loss': loss,
                'cam': cam.cpu().detach()}

    def training_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        loss_fn = 'crossent_map' if self.config.get('map_loss_factor', 0.0) > 0 else 'crossent'
        log = {f'{loss_fn}_loss': outputs['loss'].item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        predictions = outputs['p'].argmax(dim=-1)
        df_pred = {'target': batch[self.target_key],
                   'predictions': predictions.cpu().detach().numpy(),
                   pipelines.CLIP_PATH_KEY: batch[pipelines.CLIP_PATH_KEY],
                   pipelines.CLASS_NAME: batch[pipelines.CLASS_NAME]}
        df_pred = pd.DataFrame(df_pred)
        df_pred['is_correct'] = df_pred['target'] == df_pred['predictions']
        self.df_predictions.append(df_pred)
        log = {'batch_error_rate': 1 - df_pred['is_correct'].mean()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        # Save predictions
        if self.val_steps % self.config.get('val_clip_steps', 1000) == 0:
            sav_dir = os.path.join(self.val_pred_dir, 'iter_{:04d}'.format(self.val_steps))
            os.makedirs(sav_dir, exist_ok=True)
            batch_cams = outputs['cam'][torch.arange(outputs['cam'].size(0)), :, predictions]
            clip_size = batch[self.video_key][0][0].shape[:2] \
                            + (len(batch[self.video_key][0]),)
            heatmaps = heats_cube(batch_cams,
                                  cube_size=(224, 224, 32),  # Vivit 3D input clip size
                                  cubelet_size=(16, 16, 2),  # Vivit base tubelet size
                                  target_size=clip_size)
            for hm, clip, vid in zip(heatmaps, batch[self.video_key], batch['id']):
                blended_heatmap = video_utils.video_alpha_blending(hm.detach().cpu().numpy(), clip)
                video_utils.save3d(os.path.join(sav_dir, vid + '.mp4'), blended_heatmap)
        self.val_steps += 1
        return log['batch_error_rate']

    def on_validation_epoch_end(self):
        df_pred = pd.concat(self.df_predictions)
        log = {'val_error_rate': 1 - df_pred['is_correct'].mean()}
        self.log_dict(log, batch_size=self.config['batch_size'], prog_bar=True)
        output_path = self.config['val_prediction_fstr'].format(epoch=self.epoch)
        df_pred.to_csv(output_path, index=False)
        self.df_predictions = []
        self.epoch +=1
        return log['val_error_rate']

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []

    def blended_cam(self, clip_path):
        vf = pims.Video(clip_path)
        indices = video_utils.sample_frame_indices(
            clip_len=self.config['clip_len'],
            frame_sample_rate=self.config['frame_sample_rate'],
            seg_len=len(vf))
        batch = {self.video_key: [[np.array(vf[i]) for i in indices]]}
        outputs = self(batch, compute_loss=False)
        predictions = outputs['p'].argmax(dim=1).detach().cpu()
        cam = outputs.pop('cam').detach().cpu()
        batch_cams = cam[torch.arange(cam.size(0)), :, predictions]
        clip_size = batch[self.video_key][0][0].shape[:2] + (len(batch[self.video_key][0]),)
        hm = heats_cube(batch_cams,
                        cube_size=(224, 224, 32),  # Vivit 3D input clip size
                        cubelet_size=(16, 16, 2),  # Vivit base tubelet size
                        target_size=clip_size)[0]
        blended_heatmap = video_utils.video_alpha_blending(hm.numpy(), batch[self.video_key][0])
        return blended_heatmap