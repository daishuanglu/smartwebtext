from abc import ABC
import pytorch_lightning as ptl
from typing import Any, List

import numpy as np
import PIL.Image as PilImage
import re
import os
import itertools
import random
from utils.train_utils import device
from utils import color_utils
from utils import image_utils
from modules.layers import *
from thin_plate_spline_motion_model.modules.util import *
from thin_plate_spline_motion_model.modules import keypoint_detector, bg_motion_predictor

def flatten_list(nested_list):
    flattened_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened_list.extend(flatten_list(sublist))
        else:
            flattened_list.append(sublist)
    return flattened_list


class SegmentationEngine(ptl.LightningModule, ABC):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 predictions_fstr=None,
                 predictions_fsep=None,
                 label_colors_json=None):
        super(SegmentationEngine, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.bsize = batch_size
        self.lr = learning_rate
        print('Initializing Pix2pix image segment color generator.')
        self.predictions_fstr = predictions_fstr
        self.sep = predictions_fsep if predictions_fsep else None
        self.check_annotations = False
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
        self.val_optimizer_idx = 1
        self.loss_names = None

    def build_network(self):
        assert NotImplementedError(), 'Need to implement the network architecture.'

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assert NotImplementedError(), 'Need to implement the network forward.'

    def training_step(self, batch, batch_nb):
        loss_names = ['loss_%d' % i for i in range(
            len(self.optimizers()))] if self.loss_names is None else self.loss_names
        for i, (optimizer_i, loss_name) in enumerate(zip(self.optimizers(), loss_names)):
            self.toggle_optimizer(optimizer_i, optimizer_idx=i)
            _, _, loss = self.forward(batch, True, i)
            self.log(loss_name, loss.item(), prog_bar=True)
            self.manual_backward(loss)
            optimizer_i.step()
            optimizer_i.zero_grad()
            self.untoggle_optimizer(optimizer_i)

    def reg_training_step(self, batch, batch_nb):
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
        outputs, logits, loss = self(batch, compute_loss=True, optimizer_idx=self.val_optimizer_idx)
        outputs = logits if self.check_annotations else outputs
        loss = loss.item()
        if self.predictions_fstr is not None:
            predictions_paths = self.get_prediction_path(batch)
            for output, save_path in zip(outputs, predictions_paths):
                if self.check_annotations:
                    color_segments = self.label_colors[output.permute(1,2,0).argmax(axis=-1)]
                    color_segments = color_segments.detach().cpu().numpy()
                else:
                    color_segments = (output * 255).long().permute(1, 2, 0).detach().cpu().numpy()
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                PilImage.fromarray(color_segments.astype('uint8')).save(save_path)
        self.log_dict({'val_loss': loss}, batch_size=self.bsize, on_step=True, prog_bar=True)
        return {'loss': loss}

    def validation_epoch_end(self, *args, **kwargs):
        self.train()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.lr)], []

    def segments(self, list_of_images: List[np.array]):
        self.eval()
        outputs, _ = self({'frames': [torch.from_numpy(img) for img in list_of_images]})
        segments = [(output*255).long().permute(1, 2, 0).detach().numpy() for output in outputs]
        return segments.astype('uint8')


class Pix2Pix(SegmentationEngine):
    def __init__(self, config, multi_fname_sep=None):
        super(Pix2Pix, self).__init__(
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            predictions_fstr=config.get('val_prediction_fstr', None),
            predictions_fsep=multi_fname_sep,
            label_colors_json=config.get('label_colors_json', None)
        )
        self.config = config
        self.build_network()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.val_optimizer_idx = 1

    def build_network(self):
        gen_in_channels = self.config['in_channels']
        gen_out_channels = self.config['in_channels'] + self.config['out_channels']
        self.gen = Generator(
            gen_in_channels,
            features=self.config['gen_features'],
            n_layers=self.config['n_gen_enc'],
            out_channels=gen_out_channels)
        #self.gen = Hourglass(
        #    block_expansion=self.config['out_channels'],
        #    in_features=self.config['in_channels'],
        #    num_blocks=3,
        #    max_features=256)
        disc_input_channel = self.config['in_channels'] * 2
        disc_output_channel = self.config['in_channels']
        self.disc = Discriminator(
            disc_input_channel,
            self.config['disc_features'],
            disc_output_channel)

    def configure_optimizers(self):
        b1, b2 = 0.5, 0.999
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(b1, b2))
        opt_a = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(b1, b2))
        # Use the below lines if trying with annotation loss
        self.loss_names = ['d_loss', 'g_loss', 'a_loss']
        return [opt_d, opt_g, opt_a], []
        # Use the commented lines if not doing annotation loss
        #self.loss_names = ['d_loss', 'g_loss']
        #return [opt_d, opt_g], []

    def forward(self, batch, compute_loss=False, optimizer_idx=0):
        x = [f.permute(2, 0, 1) / 255.0 for f in flatten_list(batch['frames'])]
        n_frames = len(x)
        x_patches, x_coords = image_utils.image_tensors_to_patches(x, self.config['patch_size'])
        x_patches = x_patches.to(device).float()
        y_fake_logits = self.gen(x_patches)
        y_fake_logits = image_utils.patches_to_image_tensors(y_fake_logits, x_coords)
        y_fake_probs = [torch.tanh(logits) for logits in y_fake_logits]
        y_preds_logits = [y_fake[self.config['in_channels']:, :, :] for y_fake in y_fake_logits]
        y_fake_probs = [y_fake[:self.config['in_channels'], :, :] for y_fake in y_fake_probs]
        #y_preds = image_utils.patches_to_image_tensors(y_preds, x_coords)
        #y_fake = image_utils.patches_to_image_tensors(y_fake, x_coords)
        if compute_loss:
            gts =  flatten_list(batch['gt_frames'])
            gt_imgs = [self.label_colors[self.color_index[gt.to(device)]] for gt in gts]
            gt_imgs = [gt_img.permute(2,0,1).float()/255 for gt_img in gt_imgs]
            gt_index = [self.color_index[gt.to(device)] for gt in gts]
            #gt_onehot = [F.one_hot(
            #    gt_idx, num_classes=self.config['out_channels']) for gt_idx in gt_index]
            #gt_onehot = [_gt.permute(2, 0, 1).float() for _gt in gt_onehot]
            if optimizer_idx == 0:
                D_loss = 0
                for i in range(n_frames):
                    x[i] = x[i].to(device)
                    D_real = self.disc(x[i].unsqueeze(0), gt_imgs[i].unsqueeze(0))
                    D_fake = self.disc(x[i].unsqueeze(0), y_fake_probs[i].detach().unsqueeze(0))
                    D_real_loss = self.mse(D_real, torch.ones_like(D_real))
                    D_fake_loss = self.mse(D_fake, torch.zeros_like(D_fake))
                    D_loss += (D_real_loss + D_fake_loss) / 2
                D_loss /= n_frames
                return y_fake_probs, y_preds_logits, D_loss
            if optimizer_idx == 1:
                # generator loss
                G_loss = 0
                for i in range(n_frames):
                    D_fake = self.disc(x[i].to(device).unsqueeze(0), y_fake_probs[i].unsqueeze(0))
                    G_fake_loss = self.mse(D_fake, torch.ones_like(D_fake))
                    L1 = self.l1(y_fake_probs[i], gt_imgs[i]) * self.config['l1_lambda']
                    G_loss += G_fake_loss + L1
                G_loss /= n_frames
                return y_fake_probs, y_preds_logits, G_loss
            if optimizer_idx == 2:
                G_loss = 0
                for i in range(n_frames):
                    D_fake = self.disc(x[i].to(device).unsqueeze(0), y_fake_probs[i].unsqueeze(0))
                    G_fake_loss = self.mse(D_fake, torch.ones_like(D_fake))
                    y_pred_logit = y_preds_logits[i].permute(1, 2, 0).view(-1, y_preds_logits[i].shape[0])
                    L2 = self.ce(y_pred_logit, gt_index[i].reshape(-1))
                    #L2 = self.mse(y_preds[i], gt_onehot[i]) * self.config['l2_lambda']
                    G_loss += G_fake_loss + L2
                G_loss /= n_frames
                return y_fake_probs, y_preds_logits, G_loss
        else:
            return y_fake_probs,  None


class Pix2PixMotion(SegmentationEngine):
    def __init__(self, config, multi_fname_sep=None):
        super(Pix2PixMotion, self).__init__(
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            predictions_fstr=config.get('val_prediction_fstr', None),
            predictions_fsep=multi_fname_sep,
            label_colors_json=config.get('label_colors_json', None)
        )
        self.config = config
        self.build_network()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.val_optimizer_idx = 1

    def build_network(self):
        gen_in_channels = self.config['in_channels']
        self.gen_out_channels = self.config['in_channels'] + self.config['out_channels']
        self.gen = Generator(
            gen_in_channels,
            features=self.config['gen_features'],
            n_layers=self.config['n_gen_enc'],
            out_channels=self.gen_out_channels)
        disc_input_channel = self.config['in_channels'] * 2
        disc_output_channel = self.config['in_channels']
        self.disc = Discriminator(
            disc_input_channel,
            self.config['disc_features'],
            disc_output_channel)
        self.kp_extractor = keypoint_detector.KPDetector(
            **self.config['tps_model_params']['common_params'])
        self.bg_predictor = bg_motion_predictor.BGMotionPredictor()
        self.global_gru = GlobalHiddenUpdater(g_dim=self.gen_out_channels)
        self.num_tps = self.config['tps_model_params']['common_params']['num_tps']
        self.tps_deformed_weigthed_sum = nn.Linear(self.num_tps+1, 1)
        #self.motion = GeneratorFullModelFromConfig(
        #    self.config['tps_model_params'],
        #    self.config['tps_train_params'])
        #self.hourglass_gru = Hourglass(
        #    block_expansion=self.config['out_channels'],
        #    in_features=self.gen_out_channels*2,
        #    **self.config['hourglass_gru_params'])

    def configure_optimizers(self):
        b1, b2 = 0.5, 0.999
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(b1, b2))
        #opt_a = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(b1, b2))
        opt_m = torch.optim.Adam([
            {'params': self.gen.parameters()},
            {'params': self.global_gru.parameters()},
            {'params': self.bg_predictor.parameters()},
            {'params': self.kp_extractor.parameters()}
        ], lr=self.lr, betas=(b1, b2))
        #opt_m = torch.optim.Adam(
        #    self.motion.parameters(),
        #    lr=self.config['tps_train_params']['lr_generator'],
        #    betas=(b1, b2), weight_decay=1e-4)
        self.loss_names = ['d_loss', 'g_loss', 'm_loss']
        return [opt_d, opt_g, opt_m], []

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):
        # K TPS transformaions
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving['fg_kp']
        kp_2 = kp_source['fg_kp']
        kp_1 = kp_1.view(bs, -1, 5, 2)
        kp_2 = kp_2.view(bs, -1, 5, 2)
        trans = TPS(mode = 'kp', bs = bs, kp_1 = kp_1, kp_2 = kp_2)
        driving_to_source = trans.transform_frame(source_image)

        identity_grid = make_coordinate_grid((h, w), type=kp_1.type()).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # affine background transformation
        if not (bg_param is None):
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)

        transformations = torch.cat([identity_grid, driving_to_source], dim=1)
        return transformations

    def create_deformed_source_image(self, source_image, transformations):

        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_tps + 1), -1, h, w)
        transformations = transformations.view((bs * (self.num_tps + 1), h, w, -1))
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        deformed = deformed.view((bs, self.num_tps+1, -1, h, w))
        return deformed

    def disc_loss(self, x, y_fake_probs, gt_imgs, n_frames):
        D_loss = 0
        for i in range(n_frames):
            x[i] = x[i].to(device)
            D_real = self.disc(x[i].unsqueeze(0), gt_imgs[i].unsqueeze(0))
            D_fake = self.disc(x[i].unsqueeze(0), y_fake_probs[i].detach().unsqueeze(0))
            D_real_loss = self.mse(D_real, torch.ones_like(D_real))
            D_fake_loss = self.mse(D_fake, torch.zeros_like(D_fake))
            D_loss += (D_real_loss + D_fake_loss) / 2
        D_loss /= n_frames
        return D_loss

    def gen_loss(self, x, y_fake_probs, gt_imgs, n_frames):
        # generator loss
        G_loss = 0
        for i in range(n_frames):
            D_fake = self.disc(x[i].to(device).unsqueeze(0), y_fake_probs[i].unsqueeze(0))
            G_fake_loss = self.mse(D_fake, torch.ones_like(D_fake))
            L1 = self.l1(y_fake_probs[i], gt_imgs[i]) * self.config['l1_lambda']
            G_loss += G_fake_loss + L1
        G_loss /= n_frames
        return G_loss

    def annotation_loss(self, x, y_fake_probs, y_preds_logits, gt_index, n_frames):
        G_loss = 0
        for i in range(n_frames):
            D_fake = self.disc(x[i].to(device).unsqueeze(0), y_fake_probs[i].unsqueeze(0))
            G_fake_loss = self.mse(D_fake, torch.ones_like(D_fake))
            y_pred_logit = y_preds_logits[i].permute(1, 2, 0).view(-1, y_preds_logits[i].shape[0])
            L2 = self.ce(y_pred_logit, gt_index[i].reshape(-1))
            # L2 = self.mse(y_preds[i], gt_onehot[i]) * self.config['l2_lambda']
            G_loss += G_fake_loss + L2
        G_loss /= n_frames
        return G_loss

    def frame_predictions(self, x):
        x_patches, x_coords = image_utils.image_tensors_to_patches(x, self.config['patch_size'])
        x_patches = x_patches.to(device).float()
        y_fake_logits = self.gen(x_patches)
        y_fake_logits = image_utils.patches_to_image_tensors(y_fake_logits, x_coords)
        #y_fake_probs = [torch.tanh(logits) for logits in y_fake_logits]
        #y_preds_logits = [y_fake[self.config['in_channels']:, :, :] for y_fake in y_fake_logits]
        #y_fake_probs = [y_fake[:self.config['in_channels'], :, :] for y_fake in y_fake_probs]
        return y_fake_logits

    def get_driving_ids(self, batch, compute_loss):
        n_driving_ids = [len(batch_frames) for batch_frames in batch['frames']]
        driving_ids = [list(range(n)) for n in n_driving_ids]
        ni_driving_ids = np.repeat(np.cumsum([0]+n_driving_ids[:-1]), n_driving_ids)
        if compute_loss:
            # shuffle frames for training
            for ids in driving_ids:
                random.shuffle(ids)
            driving_ids = np.array(flatten_list(driving_ids)) + ni_driving_ids
            return driving_ids
        else:
            # 1-step rotate frames for testing, maybe all use first frame is ok?
            return ni_driving_ids

    def deform(self, source, driving):
        source_image = source[:self.config['in_channels'], :, :].unsqueeze(0)
        driving_image = driving[:self.config['in_channels'], :, :].unsqueeze(0)
        kp_source = self.kp_extractor(source_image)
        kp_driving = self.kp_extractor(driving_image)
        bg_param = self.bg_predictor(source_image, driving_image)
        #heatmap_representation = self.create_heatmap_representations(source_image, kp_driving, kp_source)
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)
        deformed_source = self.create_deformed_source_image(source.unsqueeze(0), transformations)
        deformed_source = deformed_source.permute(0, 2,3,4,1)
        output_deformed_source = self.tps_deformed_weigthed_sum(deformed_source)
        return output_deformed_source.squeeze(-1)

    def forward(self, batch, compute_loss=False, optimizer_idx=0):
        x = [f.permute(2, 0, 1) / 255.0 for f in flatten_list(batch['frames'])]
        driving_ids= self.get_driving_ids(batch, compute_loss)
        x_driving = [x[i] for i in driving_ids]
        if not compute_loss:
            x, x_driving = x_driving, x
        n_frames = len(x)
        y_fake_logits = self.frame_predictions(x)
        y_fake_driving_logits = self.frame_predictions(x_driving)
        y_preds_deformed_logits = [self.deform(
                y_pred, drv) for y_pred, drv in zip(y_fake_logits, x_driving)]
        y_fake_driving_logits_gated = [self.global_gru(
            deformed_logits, drv_logits.unsqueeze(0)) for deformed_logits, drv_logits in zip(
                y_preds_deformed_logits, y_fake_driving_logits)]
        # Produce source and driving image segmentation color generation probability
        y_fake_probs = [torch.tanh(logits[:self.config['in_channels'], :, :]) for logits in y_fake_logits]
        y_fake_driving_probs = [torch.tanh(
            logits[:self.config['in_channels'], :, :]) for logits in y_fake_driving_logits]
        # Produce driving image segmentation color generation probability
        y_fake_driving_probs_gated = [torch.tanh(logits.squeeze(
            0)[:self.config['in_channels'], :, :]) for logits in y_fake_driving_logits_gated]
        loss = None
        if compute_loss:
            # extract gt
            gts = flatten_list(batch['gt_frames'])
            gt_imgs = [self.label_colors[self.color_index[gt.to(device)]] for gt in gts]
            gt_imgs = [gt_img.permute(2, 0, 1).float() / 255 for gt_img in gt_imgs]
            #gt_index = [self.color_index[gt.to(device)] for gt in gts]
            gt_imgs_driving = [gt_imgs[i] for i in driving_ids]
            #gt_index_driving = [gt_index[i] for i in driving_ids]
            if optimizer_idx == 0:
                # static loss
                D_loss = self.disc_loss(x, y_fake_probs, gt_imgs, n_frames)
                D_loss += self.disc_loss(
                    x_driving, y_fake_driving_probs, gt_imgs_driving, n_frames)
                loss = D_loss / 2
            if optimizer_idx == 1:
                G_loss = self.gen_loss(x, y_fake_probs, gt_imgs, n_frames)
                G_loss += self.gen_loss(x_driving,
                                        y_fake_driving_probs,
                                        gt_imgs_driving,
                                        n_frames)
                loss = G_loss / 2
            if optimizer_idx == 2:
                M_loss = self.gen_loss(x_driving,
                                     y_fake_driving_probs_gated,
                                     gt_imgs_driving,
                                     n_frames)
                loss = M_loss
        return y_fake_driving_probs_gated, y_fake_driving_logits_gated, loss
            #if optimizer_idx == 2:
            #    M_loss = 0
            #    for src, drv in zip(gt_imgs, gt_imgs_driving):
            #        motion_loss, generated = self.motion(
            #            {'source': src.unsqueeze(0), 'driving': drv.unsqueeze(0)}, self.current_epoch)
            #        loss_values = [val.mean() for val in motion_loss.values()]
            #        M_loss += sum(loss_values)
            #    M_loss /= (n_frames *2)
            #    return y_fake_driving_probs, y_preds_driving_logits, M_loss
            #if optimizer_idx == 2:
            #    A_loss = self.annotation_loss(x, y_fake, y_preds, gt_imgs, gt_onehot, n_frames)
            #    A_loss += self.annotation_loss(x_driving,
            #                            y_fake_driving,
            #                            y_preds_driving,
            #                            gt_imgs_driving,
            #                            gt_onehot_driving,
            #                            n_frames)
            #    return y_fake, A_loss
