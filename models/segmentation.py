from PIL import Image as PIlImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
from abc import ABC
import numpy as np
import math
import scipy.io as sio
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import torchvision.transforms as transforms

TRAINING_SIZE = (256, 256)

transform = transforms.Compose([
    transforms.Resize(TRAINING_SIZE), # Resize to a fixed size
    transforms.ToTensor() # Convert PIL Image to PyTorch Tensor
])
#resize_transform = nn.Upsample(size=TRAINING_SIZE, mode='bilinear')
resize_transform = transforms.Resize(TRAINING_SIZE)

def load_bsds_image(img_path):
    img = PIlImage.open(img_path)
    img = transform(img)
    return img


def load_bsds_label(raw_label_file, region_or_edge=0):
    # region_or_edge: 0 means region map,  1 indicates edge_map
    # BSDS uses 5 human to segment each image.
    # Each human generated their own segmentation for a same image.
    mat = sio.loadmat(raw_label_file)
    human_subject = np.random.choice(5, 1, replace=True)[0]
    label_map = mat['groundTruth'][0, human_subject][0, 0][region_or_edge].astype('int64')
    label_map = resize_transform(torch.from_numpy(label_map).unsqueeze(0))
    return label_map.squeeze()

train_bsds_features = {
    'image': (lambda x: load_bsds_image(str(x))),
    'gt': (lambda x: load_bsds_label(str(x)))
}

class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        loss = self.loss(output, target)
        return loss


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_dim=False):
        super(DoubleConv, self).__init__()
        stride = 2 if reduce_dim else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # If using bilinear interpolation, use transposed convolution to
        # upsample the feature map. Otherwise, use nearest-neighbor interpolation.
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels *2, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            # Concatenate x1 and x2 along the channel dimension
            diff_h = x2.size()[2] - x1.size()[2]
            diff_w = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diff_w // 2, diff_w - diff_w // 2,
                            diff_h // 2, diff_h - diff_h // 2))
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

class UNet(ptl.LightningModule, ABC):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.config = config
        self.criterion = BCEWithLogitsLoss()
        self.in_channels = config['in_channels']
        self.features = config['features']
        # Define the encoder part of the model
        self.encoder = nn.ModuleList()
        in_features = [self.config['in_channels']]+ self.config['features']
        for i in range(len(in_features)-1):
            self.encoder.append(DoubleConv(in_features[i], in_features[i+1]))
        self.decoder = nn.ModuleList()
        out_features = [self.config['out_channels']] + self.config['features']
        for i in range(1, len(out_features)):
            self.decoder.append(Up(out_features[-i], out_features[-i-1]))
        # Define the output layer
        self.output = nn.Conv2d(
            self.config['out_channels'], self.config['out_channels'], kernel_size=1)


    def forward(self, batch, compute_loss=False):
        # Encoder
        x = torch.stack(batch['image'], dim=0)
        encoder_features = []
        for module in self.encoder:
            x = module(x)
            encoder_features.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        # Decoder
        for i, module in enumerate(self.decoder):
            x = module(x, encoder_features[-i-1])
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        if not compute_loss:
            return x, None
        else:
            target = torch.stack(batch['gt'], dim=0)
            target_one_hot = F.one_hot(target, num_classes=self.config['out_channels'])
            loss = self.criterion(x, target_one_hot.float())
            return x, loss

    def train_or_val_step(self, batch, training):
        outputs, loss = self.forward(batch, compute_loss=True)
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
