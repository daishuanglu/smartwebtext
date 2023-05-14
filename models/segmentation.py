from abc import ABC
import pytorch_lightning as ptl
import itertools
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


from utils.train_utils import device
from utils import color_utils


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


class SegmentationEngine(ptl.LightningModule, ABC):
    def __init__(self, batch_size, learning_rate):
        super(SegmentationEngine, self).__init__()
        self.criterion = BCEWithLogitsLoss()
        self.bsize = batch_size
        self.lr = learning_rate

    def build_network(self):
        assert NotImplementedError(), 'Need to implement the network architecture.'

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assert NotImplementedError(), 'Need to implement the network forward.'

    def train_or_val_step(self, batch, training):
        outputs, loss = self.forward(batch, compute_loss=True)
        loss_type = 'train' if training else 'val'
        log = {
            f'{loss_type}_loss': loss.item()
        }
        self.log_dict(log, batch_size=self.bsize, on_step=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_nb):
        return {'loss': self.train_or_val_step(batch, True)}

    def validation_step(self, batch, batch_nb):
        self.train_or_val_step(batch, False)

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.lr)], []


class UNet(SegmentationEngine):
    def __init__(self, config):
        super(UNet, self).__init__(config['batch_size'], config['learning_rate'])
        self.config = config
        self.build_network()
        if not config.get('label_colors_json', None):
            label_colors = color_utils.generate_colors(self.config['out_channels'])
            self.label_colors = torch.tensor(label_colors)
        else:
            label_colors = color_utils.load_color_codes(config['label_colors_json'])
            self.label_colors = torch.tensor([d['color'] for d in label_colors])

    def build_network(self):
        self.centercrop_transform = transforms.CenterCrop(self.config['input_size'])
        self.in_channels = self.config['in_channels']
        self.features = self.config['features']
        # Define the encoder part of the model
        self.encoder = nn.ModuleList()
        in_features = [self.config['in_channels']] + self.config['features']
        for i in range(len(in_features) - 1):
            self.encoder.append(DoubleConv(in_features[i], in_features[i + 1]))
        self.decoder = nn.ModuleList()
        out_features = [self.config['out_channels']] + self.config['features']
        for i in range(1, len(out_features)):
            self.decoder.append(Up(out_features[-i], out_features[-i - 1]))
        # Define the output layer
        self.output = nn.Conv2d(
            self.config['out_channels'], self.config['out_channels'], kernel_size=1)

    def pad_or_crop(self, list_of_tensor2d):
        padded = []
        for tensor in list_of_tensor2d:
            height_pad = max(self.config['input_size'][0] - tensor.shape[0], 0)
            width_pad = max(self.config['input_size'][1] - tensor.shape[1], 0)
            top_pad = height_pad // 2
            bottom_pad = height_pad - top_pad
            left_pad = width_pad // 2
            right_pad = width_pad - left_pad
            padded_tensor = F.pad(
                tensor, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=-1)
            padded_tensor = self.centercrop_transform(padded_tensor)
            padded.append(padded_tensor)
        return padded

    def forward(self, batch, compute_loss=False):
        x = list(itertools.chain(*batch['image']))
        x = torch.stack(x, dim=0).to(device)
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
            gt = list(itertools.chain(*batch['gt']))
            target = torch.stack(self.pad_or_crop(gt), dim=0).to(device)
            targets = target[target > -1]
            targets = torch.minimum(torch.tensor(self.config['out_channels']-1), targets)
            targets_one_hot = F.one_hot(targets, num_classes=self.config['out_channels'])
            loss = self.criterion(x[target > -1, :], targets_one_hot.float())
            return x, loss

    def segments(self, list_of_images):
        self.eval()
        x, _ = self({'image': list_of_images})
        x = x.argmax(axis=-1)
        # Use gather to assign colors to each pixel location
        color_labels = self.label_colors[x]
        color_labels = color_labels.detach().cpu().numpy()
        segments = [color_label.astype('uint8') for color_label in color_labels]
        return segments


class LFCN(SegmentationEngine):
    def __init__(self, config):
        super(LFCN, self).__init__(config['batch_size'], config['learning_rate'])
        self.config = config
        self.build_network()
        self.criterion = nn.CrossEntropyLoss()

    def build_network(self):
        TRAINING_SIZE = (512, 512)
        h, w = TRAINING_SIZE[0], TRAINING_SIZE[1]
        self.fc1 = nn.Linear(self.config['in_channels'] * h * w, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.config['num_classes'] * 32 * 32)
        self.deconv1 = nn.ConvTranspose2d(
            self.config['num_classes'], self.config['num_classes'], kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(
            self.config['num_classes'], self.config['num_classes'], kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            self.config['num_classes'], self.config['num_classes'], kernel_size=4, stride=2, padding=1)

    def forward(self, batch, compute_loss):
        x = torch.stack(batch['image'], dim=0).to(device)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, self.config['num_classes'], 32, 32)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        if not compute_loss:
            return x, None
        else:
            target = torch.stack(batch['gt'], dim=0).to(device)
            target = torch.minimum(torch.tensor(self.config['num_classes']-1), target)
            loss = self.criterion(x, target.long())
            return x, loss