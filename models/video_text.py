import torch
import pytorch_lightning as ptl
from abc import ABC
import pims
import numpy as np
from typing import Dict
from preprocessors import pipelines
from transformers import VisionEncoderDecoderModel, BertTokenizer
from transformers import AutoImageProcessor


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def load_video(vf):
    v = pims.Video(vf)
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=len(v))
    video = np.stack([v[i] for i in indices])
    return indices, video


train_kth_nobbox_features = {
    'video': (lambda x: load_video(str(x))),
    'pid': (lambda x: int(x)),
    'action': (lambda x: pipelines.KTH_ACTIONS.index(x)),
    'fids': (lambda x: list(map(int, x.split(';'))))
}


class STEncoderDecoder(ptl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vis_encdec = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path="MCG-NJU/videomae-base",
            decoder_pretrained_model_name_or_path="bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #self.vis_encdec.config.decoder_start_token_id = self.tokenizer.cls_token_id
        #self.vis_encdec.config.pad_token_id = self.tokenizer.pad_token_id
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        #self.vid_mae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    def forward(self, batch: Dict[str, torch.Tensor], compute_loss=False):
        batch_sampled_fids, batch_video = list(zip(*batch['video']))
        inputs = self.image_processor(list(map(list, batch_video)), return_tensors="pt")
        # The 1568 latent dimension can be seen as a tokenization of 16 frames, each with 98 tokens.
        inputs['decoder_input_ids'] = torch.tensor([list(range(
            self.config['max_output_seq_len'])) for _ in range(self.config['batch_size'])]).long()
        if compute_loss:
            # Use '{person} {action}' as a short caption of a frame.
            batch_pid_aid = [[
                '{person:d} {action:d}'.format(person=pid, action=aid) if fid in action_fids else 'none none'
                for fid in sampled_fids] for action_fids, sampled_fids, pid, aid in zip(*[
                batch['fids'], batch_sampled_fids, batch['pid'], batch['action']]) ]
            # Use [SEP] to separate different frames.
            batch_pid_aid_str =  ['[SEP]'.join(s) for s in batch_pid_aid]
            inputs['labels'] = self.tokenizer(
                batch_pid_aid_str,
                padding="max_length",
                truncation=True,
                max_length=self.config['max_output_seq_len'],
                return_tensors="pt").input_ids
            outputs = self.vis_encdec(**inputs)
        else:
            generated_ids = self.vis_encdec.generate(**inputs)
            outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs

    def train_or_val_step(self, batch, training):
        outputs = self.forward(batch, compute_loss=True)
        loss = outputs.loss
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, batch_size=self.config['batch_size'], on_step=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.config['learning_rate'])], []
