import torch
import pytorch_lightning as ptl
from abc import ABC
from transformers import DetrForObjectDetection, DetrImageProcessor
import pims

from preprocessors import pipelines


detr_img_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

DETR_LABEL_EXAMPLE_DICT = {
    'size': torch.tensor([800, 1066]),
    'image_id':  torch.tensor([13]),
    'class_labels':  torch.tensor([0, 0]),
    'boxes':  torch.tensor([[0.5603, 0.3812, 0.0649, 0.1152],
         [0.4990, 0.3880, 0.0742, 0.1146]]),
    'area':  torch.tensor([923.7658, 552.3618]),
    'iscrowd':  torch.tensor([0, 0]),
    'orig_size':  torch.tensor([1536, 2048])}

train_kth_features = {
    'video_path': (lambda x: str(x)),
    'x': (lambda x: float(x)),
    'y': (lambda x: float(x)),
    'w': (lambda x: float(x)),
    'h': (lambda x: float(x)),
    'action': (lambda x: pipelines.KTH_ACTIONS.index(x))
}


def detr_inputs(feature):
    v = pims.Video(feature['video_path'])
    pixel_values = v[feature['fid']]
    encoding = detr_img_processor.pad(pixel_values, return_tensors="pt")
    encoded_pixel_values = encoding['pixel_values']
    encoded_pixel_mask = encoding['pixel_mask']
    return encoded_pixel_values, encoded_pixel_mask


def detr_labels(feature):
    label = DETR_LABEL_EXAMPLE_DICT.copy()
    label['class_labels'] = torch.tensor([feature['action']])
    label['orig_size'] = torch.tensor([feature['img_height'], feature['img_width']])
    boxes = [[feature['x'], feature['y'], feature['w'], feature['h']]]
    label['boxes'] = torch.tensor(boxes)
    label['image_id'] = torch.tensor([0])
    label['size'] = label['orig_size']
    label['is_crowd'] = torch.tensor([0])
    label['area'] = torch.tensor([w*h for _,_,w,h in boxes])
    return label

kth_detr_col_fns = {'inputs': detr_inputs, 'label': detr_labels}


class Detr(ptl.LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            revision="no_timm",
                                                            num_labels=config['num_labels'],
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = config['lr']
        self.lr_backbone = config['lr_backbone']
        self.weight_decay = config['weight_decay']

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        # code refering to
        # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master
        # /DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb#scrollTo=jJjrd5vp2PWe
        # pixel_values = batch["pixel_values"]
        # pixel_mask = batch["pixel_mask"]
        # unwrap pixel value and masks
        pixel_values, pixel_mask = list(zip(*batch['inputs']))
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["label"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("train_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        for k, v in loss_dict.items():
            self.log("val_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer, []