import os
from preprocessors import pipelines
import pandas as pd
import torch
import pytorch_lightning as ptl
from abc import ABC
from typing import Dict
import pims
import numpy as np
from p_tqdm import p_map
from transformers import VivitImageProcessor, VivitModel, AutoImageProcessor, ViTModel
from transformers import AutoFeatureExtractor, SwinModel, Swinv2Model

import torch.nn as nn
import torch.nn.functional as F

from modules import omp
from utils import video_utils, train_utils, metric_utils

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    nominator = torch.sum(token_embeddings * input_mask_expanded, 1)
    denominator = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return nominator / denominator


def heats_cube(heats, cubelet_size, cube_size, target_size=None):
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


def blended_cam(model,
                clip_path,
                clip_len,
                frame_sample_rate,
                cubelet_size=(16, 16, 2)):
    vf = pims.Video(clip_path)
    indices = video_utils.sample_frame_indices(
            clip_len=clip_len,
            frame_sample_rate=frame_sample_rate,
            seg_len=len(vf))
    batch = {model.video_key: [[np.array(vf[i]) for i in indices]]}
    outputs = model(batch)
    predictions = outputs['p'].argmax(dim=1).detach().cpu()
    cam = outputs.pop('cam').detach().cpu()
    batch_cams = cam[torch.arange(cam.size(0)), :, predictions]
    clip_size = batch[model.video_key][0][0].shape[:2] + (len(batch[model.video_key][0]),)
    hm = heats_cube(batch_cams,
                    cube_size=(224, 224, 32),  # Vivit 3D input clip size
                    cubelet_size=cubelet_size,  # Vivit base tubelet size
                    target_size=clip_size)[0]
    blended_heatmap = video_utils.video_alpha_blending(hm.numpy(), batch[model.video_key][0])
    return blended_heatmap, hm, outputs


class VideoRecognitionEngine(ptl.LightningModule, ABC):
    def __init__(self, config, video_key, target_key, clsname_map, **kwargs):
        super().__init__()
        self.video_key = video_key
        self.target_key = target_key
        self.num_classes = len(clsname_map)
        self.config = config
        self.image_processor = None
        self.model = None
        self.ce = nn.CrossEntropyLoss()
        self.clsname_map = clsname_map
        self.bce = nn.BCEWithLogitsLoss()
        self.df_predictions = []
        self.epoch = 0
        self.val_cam = None
        self.val_steps = 0
        self.cubelet_size = self.config['cubelet_size']
        if self.config.get('val_prediction_fstr', ''):
            self.val_pred_dir = os.path.dirname(self.config['val_prediction_fstr'])
            os.makedirs(self.val_pred_dir, exist_ok=True)

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        raise NotImplementedError('video Recognition model is not implemented.')

    def training_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        loss_name = 'crossent' if self.config.get('mulitclass', '') else 'BCELoss'
        loss_fn = f'{loss_name}_map' if self.config.get('map_loss_factor', 0.0) > 0 else loss_name
        log = {f'{loss_fn}_loss': outputs['loss'].item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        if self.config.get('multiclass', False):
            predictions = [torch.nonzero(row > 0.5).squeeze().tolist() for row in outputs['p']]
            cam_ind = [p[0] for p in predictions]
        else:
            predictions = outputs['p'].argmax(dim=-1).cpu().detach().numpy()
            cam_ind = predictions
        df_pred = {'target': batch[self.target_key],
                   'predictions': predictions,
                   pipelines.CLIP_PATH_KEY: batch[pipelines.CLIP_PATH_KEY],
                   pipelines.CLASS_NAME: batch[pipelines.CLASS_NAME]}
        df_pred = pd.DataFrame(df_pred)
        if self.config.get('multiclass', False):
            df_pred['recall'] = df_pred.apply(
                lambda x: sum(p in x['target']
                              for p in x['predictions'])/len(x['target']), axis=1)
            df_pred['is_correct'] = df_pred.apply(
                lambda x: sum(p in x['target']
                              for p in x['predictions']) / len(x['predictions']), axis=1)
        else:
            df_pred['is_correct'] = df_pred['target'] == df_pred['predictions']
        self.df_predictions.append(df_pred)
        log = {'batch_error_rate': 1 - df_pred['is_correct'].mean()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        # Save predictions
        if self.val_steps % self.config.get('val_clip_steps', 1000) == 0:
            sav_dir = os.path.join(self.val_pred_dir, 'iter_{:04d}'.format(self.val_steps))
            os.makedirs(sav_dir, exist_ok=True)
            batch_cams = outputs['cam'][torch.arange(outputs['cam'].size(0)), :, cam_ind]
            clip_size = batch[self.video_key][0][0].shape[:2] \
                            + (len(batch[self.video_key][0]),)
            heatmaps = heats_cube(batch_cams,
                                  cube_size=(224, 224, 32),  # Vivit 3D input clip size
                                  cubelet_size=self.cubelet_size,  # Vivit base tubelet size
                                  target_size=clip_size)
            for hm, clip, vid in zip(heatmaps, batch[self.video_key], batch[pipelines.SAMPLE_ID_KEY]):
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


class VivitRecgModel(VideoRecognitionEngine):
    def __init__(self, config, video_key, target_key, clsname_map, **kwargs):
        super().__init__(config, video_key, target_key, clsname_map)
        self.image_processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.dense_pool_emb = nn.Linear(768, self.num_classes)

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
        cam = self.dense_pool_emb(p_encoded_seq)[:, 1:]
        loss = None
        if compute_loss:
            if self.config.get('multiclass', False):
                nrows = len(batch[self.target_key])
                targets = torch.zeros(nrows, self.num_classes)
                inds = [[(r, c) for c in cols] for r, cols in enumerate(batch[self.target_key])]
                inds = sum(inds, [])
                rows, cols = zip(*inds)
                targets[rows, cols] = 1.0
                targets = targets.to(train_utils.device).float()
                loss = self.bce(logits, targets)
                p = torch.sigmoid(logits)
            else:
                targets = torch.tensor(batch[self.target_key]).to(train_utils.device)
                loss = self.ce(logits, targets.long())
                p = F.softmax(logits, dim=-1)
        return {'p': p.cpu().detach(),
                'loss': loss,
                'cam': cam.cpu().detach()}


class VitRecgModel(VideoRecognitionEngine):
    def __init__(self, config, video_key, target_key, clsname_map, **kwargs):
        super().__init__(config, video_key, target_key, clsname_map)
        hf_pretrained_model_name = 'google/vit-base-patch16-224-in21k'
        self.image_processor = AutoImageProcessor.from_pretrained(hf_pretrained_model_name)
        self.model = ViTModel.from_pretrained(hf_pretrained_model_name)
        self.dense_pool_emb = nn.Linear(768, self.num_classes)

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        batch_videos = sum(batch[self.video_key], [])
        inputs = self.image_processor(batch_videos, return_tensors="pt")
        inputs = inputs.to(train_utils.device)
        outputs = self.model(**inputs)
        # num_of_frames * 197 (16*16 + 1) * embed_size (768)
        encoded_seq = outputs.last_hidden_state
        p_encoded_seq = torch.sigmoid(encoded_seq)
        #num_of_frames * embed_size (768)
        pooled_seq = torch.mean(p_encoded_seq, dim=1)
        # num_of_frames * num_classes
        logits = self.dense_pool_emb(pooled_seq)
        # batch_size * clip_len * num_classes
        p = F.softmax(logits, dim=-1)
        # batch_size * num_classes
        p = p.view((-1, self.config['clip_len']) + p.shape[1:]).mean(1)
        # num_of_frames * 196 * num_classes
        cam = self.dense_pool_emb(p_encoded_seq)[:, 1:]
        cam = cam.view((-1, self.config['clip_len']) + cam.shape[1:])
        nrows, ncols =  224 // self.cubelet_size[0], 224 // self.cubelet_size[0]
        cam = cam.reshape(-1, int(self.config['clip_len']* nrows*ncols), self.num_classes)
        loss = None
        if compute_loss:
            targets = torch.tensor(batch[self.target_key]).to(train_utils.device)
            targets = targets.unsqueeze(1).repeat(1, self.config['clip_len']).view(-1)
            loss = self.ce(logits, targets.long())
        return {'p': p.cpu().detach(),
                'loss': loss,
                'cam': cam.cpu().detach()}


class SwinRecgModel(VideoRecognitionEngine):
    def __init__(self, config, video_key, target_key, clsname_map, **kwargs):
        super().__init__(config, video_key, target_key, clsname_map)
        self.image_processor = AutoFeatureExtractor.from_pretrained(
            "microsoft/swin-base-patch4-window7-224")
        self.model = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224")
        self.dense_pool_emb = nn.Linear(1024, self.num_classes)

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        batch_videos = sum(batch[self.video_key], [])
        inputs = self.image_processor(batch_videos, return_tensors="pt")
        inputs = inputs.to(train_utils.device)
        outputs = self.model(**inputs)
        # num_of_frames * 197 (16*16 + 1) * embed_size (768)
        encoded_seq = outputs.last_hidden_state
        p_encoded_seq = torch.sigmoid(encoded_seq)
        #num_of_frames * embed_size (768)
        pooled_seq = torch.mean(p_encoded_seq, dim=1)
        # num_of_frames * num_classes
        logits = self.dense_pool_emb(pooled_seq)
        # batch_size * clip_len * num_classes
        p = F.softmax(logits, dim=-1)
        # batch_size * num_classes
        p = p.view((-1, self.config['clip_len']) + p.shape[1:]).mean(1)
        # num_of_frames * 197 * num_classes
        cam = self.dense_pool_emb(p_encoded_seq)
        # batch_size * clip_len * 197 * num_classes
        cam = cam.view((-1, self.config['clip_len']) + cam.shape[1:])
        nrows, ncols = 224 // self.cubelet_size[0], 224 // self.cubelet_size[0]
        cam = cam.reshape(-1, int(self.config['clip_len'] * nrows * ncols), self.num_classes)
        loss = None
        if compute_loss:
            targets = torch.tensor(batch[self.target_key]).to(train_utils.device)
            targets = targets.unsqueeze(1).repeat(1, self.config['clip_len']).view(-1)
            loss = self.ce(logits, targets.long())
        return {'p': p.cpu().detach(),
                'loss': loss,
                'cam': cam.cpu().detach()}


class EncodedSparseCoding(ptl.LightningModule, ABC):
    def __init__(self, config, video_key, **kwargs):
        super().__init__()
        if config['encoder_type'] == 'swin':
            self.image_processor = AutoFeatureExtractor.from_pretrained(
                    config['pretrained_model_name'])
            self.model = SwinModel.from_pretrained(config['pretrained_model_name'])
        if config['encoder_type'] == 'vit':
            self.image_processor = AutoImageProcessor.from_pretrained(
                    config['pretrained_model_name'])
            self.model = ViTModel.from_pretrained(config['pretrained_model_name'])
        if config['encoder_type'] == 'vivit':
            self.image_processor = VivitImageProcessor.from_pretrained(
                    config['pretrained_model_name'])
            self.model = VivitModel.from_pretrained(config['pretrained_model_name'])
        self.video_key = video_key
        self.config = config

    def omp(self, encoded):
        D, y = encoded[1:], encoded[0]
        x = omp.matching_pursuit(y, D,
                                 eps_min=self.config['omp_eps'],
                                 iter_max=self.config['iter_max'],
                                 verbose=self.config.get('verbose', False))
        basis_amp_hm = (x.transpose() * D).mean(1)
        return metric_utils.norm01(basis_amp_hm)

    def forward(self, batch: Dict[str, torch.Tensor]):
        if self.config['encoder_type'] != 'vivit':
            batch_videos = sum(batch[self.video_key], [])
            inputs = self.image_processor(batch_videos, return_tensors="pt")
        else:
            inputs = self.image_processor(batch[self.video_key], return_tensors="pt")
        inputs = inputs.to(train_utils.device)
        outputs = self.model(**inputs)
        # num_of_frames * 197 (16*16 + 1) * embed_size (768)
        e = torch.sigmoid(outputs['last_hidden_state'])
        e = F.normalize(e, p=2, dim=-1)
        cam = p_map(self.omp, list(e.detach().cpu().numpy()))
        cam = torch.from_numpy(np.stack(cam)).unsqueeze(-1)
        if self.config['encoder_type'] != 'vivit':
            cam = cam.view((-1, self.config['clip_len']) + cam.shape[1:], 1)
            nrows, ncols = 224 // self.config['cubelet_size'][0], 224 // self.config['cubelet_size'][0]
            cam = cam.reshape(-1, int(self.config['clip_len'] * nrows * ncols), 1)
        return {'p': torch.ones(cam.shape[0], 1),
                'loss': None,
                'cam': cam.cpu().detach()}