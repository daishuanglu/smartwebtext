from abc import ABC
import pytorch_lightning as ptl
from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

import PIL.Image as PilImage
import re
import os
import itertools
from utils.train_utils import device
from utils import color_utils
from utils import image_utils

"""
def pad_or_crop(self, list_of_tensor2d, centercrop_transform):
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
        padded_tensor = centercrop_transform(padded_tensor)
        padded.append(padded_tensor)
    return padded
"""
def flatten_list(nested_list):
    flattened_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened_list.extend(flatten_list(sublist))
        else:
            flattened_list.append(sublist)
    return flattened_list


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        loss = self.loss(output, target)
        return loss


def generate_color_segments(probs2D, label_colors):
    classId = probs2D.argmax(axis=-1)
    # Use gather to assign colors to each pixel location
    color_maps = label_colors[classId].numpy()
    return color_maps


class SegmentationEngine(ptl.LightningModule, ABC):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 predictions_fstr=None,
                 predictions_fsep=None,
                 label_colors_json=None,
                 classifier = True):
        super(SegmentationEngine, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.criterion = BCEWithLogitsLoss()
        self.bsize = batch_size
        self.lr = learning_rate
        self.classifier = classifier
        if classifier:
            print('Initializing Pix2pix image segmentor.')
        else:
            print('Initializing Pix2pix image generator.')
        self.predictions_fstr = predictions_fstr
        self.sep = predictions_fsep if predictions_fsep else None
        if not label_colors_json:
            label_colors = color_utils.generate_colors(self.config['out_channels'])
            self.label_colors = torch.tensor(label_colors)
        else:
            label_colors = color_utils.load_color_codes(label_colors_json)
            ids = [d['id'] for d in label_colors]
            self.label_colors = torch.tensor([d['color'] for d in label_colors]).to(device)
            self.color_index = -torch.ones(max(ids) + 1)
            for i, d in enumerate(label_colors):
                self.color_index[d['id']] = i
            self.color_index = self.color_index.long().to(device)

    def build_network(self):
        assert NotImplementedError(), 'Need to implement the network architecture.'

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assert NotImplementedError(), 'Need to implement the network forward.'

    def training_step(self, batch, batch_nb):
        # Train generator
        outputs, log_loss = self.forward(batch, True)
        self.log_dict({'train_loss': log_loss}, batch_size=self.bsize, on_step=True, prog_bar=True)
        return {'loss': log_loss}

    def get_prediction_path(self, batch):
        variable_keys = re.findall(r'{(.*?)}', self.predictions_fstr)
        # get path variables
        fstr_var_values = []

        for key_values in zip(*[batch[key] for key in variable_keys]):
            key_vals = [
                val.split(self.sep) if self.sep is not None else [val] for val in key_values]
            fstr_var_values += list(itertools.product(*key_vals)).copy()

        prediction_file_paths = []
        for vals in fstr_var_values:
            fvals = {k: v for k, v in zip(variable_keys, vals)}
            prediction_file_paths.append(self.predictions_fstr.format(**fvals))
        return prediction_file_paths

    def validation_step(self, batch, batch_nb, optimizer_idx=1):
        outputs, loss = self.forward(batch, compute_loss=True, optimizer_idx=optimizer_idx)
        if self.predictions_fstr is not None:
            predictions_paths = self.get_prediction_path(batch)
            n = 0
            for patch_coords, save_path in zip(sum(batch['patch_coords'], []), predictions_paths):
                patch_segments = []
                for i in range(len(patch_coords)):
                    if self.classifier:
                        classId = outputs[n].argmax(axis=-1)
                        # Use gather to assign colors to each pixel location
                        patch_segment = self.label_colors[classId].detach()
                    else:
                        patch_segment = outputs[n].detach() * 255.0
                    patch_segments.append(patch_segment.cpu().numpy())
                    n += 1
                color_segments = image_utils.patches_to_image(patch_segments, patch_coords)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                PilImage.fromarray(color_segments).save(save_path)
        self.log_dict({'val_loss': loss.item()}, batch_size=self.bsize, on_step=True, prog_bar=True)
        return {'loss': loss.item()}

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.lr)], []

    def segments(self, list_of_images):
        self.eval()
        segments = []
        for img in list_of_images:
            patches, patch_coords = image_utils.image_to_patches(
                img, self.config['patch_size'])
            x, _ = self({'image': patches})
            if self.classifier:
                classId = x.argmax(axis=-1)
                # Use gather to assign colors to each pixel location
                patch_segments = self.label_colors[classId].detach().cpu().numpy()
            else:
                patch_segments = [p.cpu().numpy() * 255.0 for p in x]
            segment = image_utils.patches_to_image(patch_segments, patch_coords)
            segments.append(segment)
        return segments


class DiscCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DiscCNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64,128, 256, 512], out_channels=1):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.layers = nn.ModuleList()
        in_channels = features[0]
        for feature in features[1:]:
            self.layers.append(
                DiscCNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        self.output = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x

class GenCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(GenCNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2) if act == 'leaky' else nn.ReLU()
        )
        self.use_dropout= use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, n_layers=6, out_channels=None):
        super(Generator, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode='reflect')
        )
        self.down_layers = nn.ModuleList()
        channel_multipliers = [min(8, 2**i) for i in range(n_layers)]
        in_multiplier = channel_multipliers[0]
        for out_multiplier in channel_multipliers[1:]:
            self.down_layers.append(GenCNNBlock(
                features*in_multiplier, features*out_multiplier, down=True, act='relu', use_dropout=False))
            in_multiplier = out_multiplier
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*in_multiplier,
                      features*in_multiplier,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'),
            nn.ReLU(), # 1x1
            GenCNNBlock(
                features*in_multiplier, features*in_multiplier, down=False, act='relu', use_dropout=False),
        )
        self.up_layers = nn.ModuleList()
        for out_multiplier in channel_multipliers[::-1][1:]:
            self.up_layers.append(GenCNNBlock(
                features * in_multiplier * 2,
                features * out_multiplier,
                down=False, act='leaky', use_dropout=False))
            in_multiplier = out_multiplier
        self.output = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        encoder_outputs = [x]
        for layer in self.down_layers:
            x = layer(x)
            encoder_outputs.append(x)
        x = self.bottleneck(x)
        for layer in self.up_layers:
            x = layer(torch.cat((x, encoder_outputs.pop()), dim=1))
        x = self.output(torch.cat((x, encoder_outputs.pop()), dim=1))
        return x


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


class Pix2Pix(SegmentationEngine):
    def __init__(self, config, multi_fname_sep=None):
        super(Pix2Pix, self).__init__(
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            predictions_fstr=config.get('val_prediction_fstr', None),
            predictions_fsep=multi_fname_sep,
            label_colors_json=config.get('label_colors_json', None),
            classifier=config.get('is_segmentor', True),
        )
        self.config = config
        self.build_network()
        if self.classifier:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def build_network(self):
        self.gen = Generator(
            self.config['in_channels'],
            features=self.config['gen_features'],
            n_layers=self.config['n_gen_enc'],
            out_channels=self.config['out_channels'] \
                if self.classifier else self.config['in_channels']
        )
        disc_input_channel = self.config['in_channels']+self.config['out_channels'] \
            if self.classifier else self.config['in_channels'] * 2
        disc_output_channel = self.config['out_channels'] \
            if self.classifier else self.config['in_channels']
        self.disc = Discriminator(
            disc_input_channel,
            self.config['disc_features'],
            disc_output_channel)

    def training_step(self, batch, batch_nb):
        optimizer_d, optimizer_g = self.optimizers()
        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d, optimizer_idx=0)
        #self.set_requires_grad(self.disc, True)  # enable backprop for D
        outputs, d_loss = self.forward(batch, True, 0)
        self.log("d_loss", d_loss.item(), prog_bar=True)
        # self.log_dict({'d_loss': d_loss}, batch_size=self.bsize, on_step=True, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        #self.set_requires_grad(self.disc, False)  # enable backprop for D
        self.toggle_optimizer(optimizer_g, optimizer_idx=1)
        outputs, g_loss = self.forward(batch, True, 1)
        self.log("g_loss", g_loss.item(), prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

    def configure_optimizers(self):
        b1, b2 = 0.5, 0.999
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(b1, b2))
        return [opt_d, opt_g], []

    def forward(self, batch, compute_loss=False, optimizer_idx=0):
        x = flatten_list(batch['image'])
        x = torch.stack(x, dim=0).to(device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        y_fake = self.gen(x)
        if compute_loss:
            gt = flatten_list(batch['gt'])
            if not self.classifier:
                gt_imgs = [self.label_colors[target.to(device)] for target in gt]
                targets = torch.stack(gt_imgs, dim=0).float().to(device) / 255.0
            else:
                targets = torch.stack(gt, dim=0).to(device)
                targets = self.color_index[targets.long()]
                targets = F.one_hot(targets, num_classes=self.config['out_channels'])
            targets = targets.permute(0, 3, 1, 2)
            if optimizer_idx == 0:
                D_real = self.disc(x, targets)
                D_fake = self.disc(x, y_fake.detach())
                D_real_loss = self.bce(D_real, torch.ones_like(D_real))
                D_fake_loss = self.bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
                return y_fake.permute(0, 2, 3, 1), D_loss
            if optimizer_idx == 1:
                # generator loss
                D_fake = self.disc(x, y_fake)
                G_fake_loss = self.bce(D_fake, torch.ones_like(D_fake))
                L1 = self.l1(y_fake, targets) * self.config['l1_lambda']
                G_loss = G_fake_loss + L1
                return y_fake.permute(0, 2, 3, 1), G_loss
        else:
            return y_fake.permute(0, 2, 3, 1), None


class UNet(SegmentationEngine):
    def __init__(self, config):
        super(UNet, self).__init__(
            config['batch_size'], config['learning_rate'], config.get('label_colors_json', None))
        self.config = config
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.build_network()

    def build_network(self):
        self.gen = Generator(
            self.config['in_channels'],
            features=self.config['gen_features'],
            n_layers=self.config['n_gen_enc'],
            out_channels=self.config['in_channels']
        )

    def forward(self, batch, compute_loss=False, **kwargs):
        x = flatten_list(batch['image'])
        x = torch.stack(x, dim=0).to(device)
        x = x.permute(0, 3, 1, 2)
        pred = self.gen(x.float())
        if compute_loss:
            gt = flatten_list(batch['gt'])
            targets = torch.stack(gt, dim=0).to(device)
            targets = torch.minimum(torch.tensor(self.config['out_channels'] - 1), targets)
            targets = F.one_hot(targets, num_classes=self.config['out_channels'])
            targets = targets.permute(0, 3, 1, 2).float()
            loss = self.l1(pred, targets) + self.bce(pred, targets)
            return pred.permute(0, 2, 3, 1), loss
        else:
            return pred, None
