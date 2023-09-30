import os
import cv2
import torch
import pandas as pd
import pims
import numpy as np

import pytorch_lightning as ptl
from abc import ABC
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import VivitImageProcessor, VivitModel
from transformers import AutoModel, AutoTokenizer
from preprocessors import pipelines
from utils import video_utils, train_utils, visual_utils


PRETRAINED_BERT_MODEL = 'bert-base-uncased'

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


class VideoTextDiscLocalization(ptl.LightningModule, ABC):
    def __init__(self, config, video_key, text_key, target_key):
        super().__init__()
        self.video_key = video_key
        self.target_key = target_key
        self.text_key = text_key
        self.config = config
        self.image_processor = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400")
        self.video_model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BERT_MODEL)
        self.text_model = AutoModel.from_pretrained(PRETRAINED_BERT_MODEL, output_attentions=True)
        self.dense_pool_emb = nn.Linear(768, 1)
        self.mse = nn.MSELoss()
        self.df_predictions = []
        self.epoch = 0
        self.val_cam = None
        self.val_steps = 0
        self.val_pred_dir = os.path.dirname(self.config['val_prediction_fstr'])
        os.makedirs(self.val_pred_dir, exist_ok=True)

    def text_encoder_input(self, list_of_string):
        encoding = self.tokenizer(list_of_string,
                                  return_tensors="pt",
                                  padding="max_length",
                                  truncation=True,
                                  max_length=self.config['max_input_length'])
        encoding = {k: v.to(train_utils.device) for k, v in encoding.items()}
        return encoding

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        # Video encoding
        inputs_video = self.image_processor(batch[self.video_key], return_tensors="pt")
        inputs_video = inputs_video.to(train_utils.device)
        outputs_video = self.video_model(**inputs_video)
        # batch_size * 3137 (16*16*16 + 1) * embed_size (768)
        encoded_seq_video = outputs_video.last_hidden_state
        p_encoded_seq_video = torch.sigmoid(encoded_seq_video)
        pooled_seq_video = torch.mean(p_encoded_seq_video, dim=1)
        # Text encoding
        inputs_text = self.text_encoder_input(batch[self.text_key])
        inputs_text['token_type_ids'] += 1
        outputs_text = self.text_model(**inputs_text)
        encoded_seq_text = outputs_text.last_hidden_state
        p_encoded_seq_text = torch.sigmoid(encoded_seq_text)
        pooled_seq_text = mean_pooling(p_encoded_seq_text, inputs_text['attention_mask'])
        # Video Text correlation
        pooled_seq = pooled_seq_text + pooled_seq_video
        logits = self.dense_pool_emb(pooled_seq)
        p = torch.sigmoid(logits).squeeze(-1)
        # Video text activation mapping
        cam_text = self.dense_pool_emb(p_encoded_seq_text)
        cam_video = self.dense_pool_emb(p_encoded_seq_video)
        loss = None
        if compute_loss:
            targets = torch.tensor(batch[self.target_key]).to(train_utils.device)
            loss = self.mse(p, targets.float())
        toks = [self.tokenizer.convert_ids_to_tokens(tok) for tok in inputs_text['input_ids']]
        return {'p': p.cpu().detach(),
                'loss': loss,
                'cam_text': cam_text.cpu().detach().squeeze(-1),
                'cam_video': cam_video.cpu().detach().squeeze(-1),
                'tokens': toks,
                'attn_mask': inputs_text['attention_mask']}

    def training_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        loss_fn = 'mse_map_loss' if self.config.get('map_loss_factor', 0.0) > 0 else 'mse_loss'
        log = {f'{loss_fn}_loss': outputs['loss'].item()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return {'loss': outputs['loss']}

    def validation_step(self, batch, batch_nb):
        outputs = self.forward(batch, compute_loss=True)
        df_pred = {'target': batch[self.target_key].cpu().detach().numpy(),
                   'predictions': outputs['p'].cpu().detach().numpy(),
                   pipelines.CLIP_PATH_KEY: batch[pipelines.CLIP_PATH_KEY],
                   pipelines.CLASS_NAME: batch[pipelines.CLASS_NAME],
                   self.text_key: batch[self.text_key]}
        df_pred = pd.DataFrame(df_pred)
        df_pred['is_correct'] = df_pred['target'] == (df_pred['predictions'] > 0.5).astype(float)
        self.df_predictions.append(df_pred)
        log = {'batch_error_rate': 1 - df_pred['is_correct'].mean()}
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        # Save predictions
        if self.val_steps % self.config.get('val_clip_steps', 1000) == 0:
            sav_dir = os.path.join(self.val_pred_dir, 'iter_{:04d}'.format(self.val_steps))
            os.makedirs(sav_dir, exist_ok=True)
            # Generate text heatmaps
            cam_text = outputs.pop('cam_text').numpy()
            end_pos_ids = outputs['attn_mask'].sum(1)
            for toks, cam, end_id, vid, score in zip(outputs['tokens'],
                                              cam_text,
                                              end_pos_ids,
                                              batch[pipelines.SAMPLE_ID_KEY],
                                              outputs['p'].numpy()):
                text_heatmap = visual_utils.text_class_activation_map(
                    toks, cam, start_end_ids=(1, end_id-1))
                cv2.imwrite(os.path.join(sav_dir, vid + '_pred{:.4f}.jpg').format(score),
                            text_heatmap)
            # Generate video heatmaps
            cam_video = outputs.pop('cam_video').detach().cpu()
            clip_size = batch[self.video_key][0][0].shape[:2] \
                        + (len(batch[self.video_key][0]),)
            heatmaps = heats_cube(cam_video,
                                  cube_size=(224, 224, 32),  # Vivit 3D input clip size
                                  cubelet_size=(16, 16, 2),  # Vivit base tubelet size
                                  target_size=clip_size)
            for hm, clip, vid, target in zip(heatmaps,
                                             batch[self.video_key],
                                             batch[pipelines.SAMPLE_ID_KEY],
                                             batch[self.target_key]):
                blended_heatmap = video_utils.video_alpha_blending(hm.detach().cpu().numpy(), clip)
                desc = 'right' if target > 0.5 else 'wrong'
                video_utils.save3d(os.path.join(sav_dir, vid + '_{:s}_desc.mp4').format(desc),
                                   blended_heatmap)
        self.val_steps += 1
        return log['batch_error_rate']

    def on_validation_epoch_end(self):
        df_pred = pd.concat(self.df_predictions)
        log = {'val_error_rate': 1 - df_pred['is_correct'].mean()}
        self.log_dict(log, batch_size=self.config['batch_size'], prog_bar=True)
        output_path = self.config['val_prediction_fstr'].format(epoch=self.epoch)
        df_pred.to_csv(output_path, index=False)
        self.df_predictions = []
        self.epoch += 1
        return log['val_error_rate']

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []

    def blended_video_text_cam(self, clip_path, text):
        # This is for inference only
        vf = pims.Video(clip_path)
        indices = video_utils.sample_frame_indices(
            clip_len=self.config['clip_len'],
            frame_sample_rate=self.config['frame_sample_rate'],
            seg_len=len(vf))
        batch = {self.video_key: [[np.array(vf[i]) for i in indices]],
                 self.text_key: [text]}
        outputs = self(batch, compute_loss=False)
        cam_video = outputs.pop('cam_video').detach().cpu()
        clip_size = batch[self.video_key][0][0].shape[:2] + (len(batch[self.video_key][0]),)
        hm = heats_cube(cam_video,
                        cube_size=(224, 224, 32),  # Vivit 3D input clip size
                        cubelet_size=(16, 16, 2),  # Vivit base tubelet size
                        target_size=clip_size)[0]
        blended_heatmap = video_utils.video_alpha_blending(hm.numpy(), batch[self.video_key][0])
        cam_text = outputs.pop('cam_text').detach().cpu()
        end_pos_ids = outputs['attn_mask'].sum(1)
        text_heatmap = visual_utils.text_class_activation_map(
                outputs['tokens'][0],  cam_text, start_end_ids=(1, end_pos_ids[0]-1))
        return blended_heatmap, text_heatmap