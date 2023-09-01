import pytorch_lightning as ptl
from abc import ABC
from transformers import ViTFeatureExtractor, ViTModel
from typing import Dict
import torch
import torch.nn as nn

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TransformerCAM(ptl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dense_mapping = nn.Linear(768, 1)
        self.mse = nn.MSELoss()

    def class_activation_mapping(self, encoded_seq):
        weights = self.dense_mapping.weight.data[0].T
        cam = torch.matmul(encoded_seq, weights)
        return cam.squeeze(-1)

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        inputs =self.vit(images=batch['pil_image'], return_tensors="pt")
        outputs = self.vit(**inputs)
        encoded_seq = outputs.last_hidden_state
        encoded_seq = torch.sigmoid(encoded_seq)
        attn_mask = torch.ones(encoded_seq.shape[:2]).to(encoded_seq.device)
        attn_mask[:, 0] = 0.0
        encoded_cls = encoded_seq[:, 0, :]
        cls_mp_seq = encoded_cls + mean_pooling(encoded_seq[:, 1:, :], attn_mask)
        proba = self.dense_mapping(cls_mp_seq)
        cam = self.class_activation_mapping(encoded_seq)
        loss = None
        if compute_loss:
            targets = encoded_cls
            cam_max = torch.max(cam, dim=-1).values
            cam_min = torch.max(cam, dim=-1).values
            loss = self.mse(proba, targets) - (targets * 2 - 1) * ((cam_max - cam_min) ** 2)
        return {'proba': proba, 'loss': loss, 'cam': cam}

    def train_or_val_step(self, batch, training):
        outputs = self.forward(batch, compute_loss=True)
        loss = outputs['loss']
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_nb):
        return {'MSEloss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []
