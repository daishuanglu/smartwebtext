from abc import ABC
import pytorch_lightning as ptl
from typing import Any, List

import numpy as np
import PIL.Image as PilImage
import re
import os
import itertools
from functools import partial
import torch
from utils.train_utils import device
from utils import color_utils, image_utils, data_utils, visual_utils
from modules import pspnet, pix2pix, losses, ops, hungarian


def pad_or_crop_obj_mask(mask, num_of_objects):
    padding_size = max(0, num_of_objects - mask.shape[0])
    return torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, padding_size))[:num_of_objects]


def local_mse(y_onehot, y_hat):
    return (y_onehot - y_hat) ** 2


def local_nll(y_onehot, y_hat):
    return y_onehot * y_hat + (1- y_onehot) * (1 - y_hat)


class SegmentationEngine(ptl.LightningModule, ABC):
    def __init__(self,
                 batch_size,
                 learning_rate=1e-5,
                 predictions_fstr=None,
                 predictions_fsep=None,
                 bg_min_confidence=0.3,
                 min_confidence=0.3,
                 max_val_viz_batch=5):
        super(SegmentationEngine, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.bsize = batch_size
        self.lr = learning_rate
        print('Initializing Pix2pix image segment color generator.')
        self.predictions_fstr = predictions_fstr
        self.sep = predictions_fsep if predictions_fsep else None
        self.check_annotations = False
        self.val_optimizer_idx = 1
        self.loss_names = None
        self.bg_conf_thresh = bg_min_confidence
        self.conf_thresh = min_confidence
        self.label_colors = []
        self.max_val_viz_batch = max_val_viz_batch

    def build_network(self):
        assert NotImplementedError(), 'Need to implement the network architecture.'

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        assert NotImplementedError(), 'Need to implement the network forward.'

    def training_step(self, batch, batch_nb):
        n_opt = len(self.optimizers()) if isinstance(self.optimizers(), list) else 1
        if not self.loss_names:
            loss_names = ['loss_%d' % i for i in range(n_opt)]
        else:
            loss_names = self.loss_names
        if n_opt == 1:
            opts = [self.optimizers()]
        else: 
            opts = self.optimizers()
        for i, (opt, loss_name) in enumerate(zip(opts, loss_names)):
            if n_opt > 1: self.toggle_optimizer(opt, optimizer_idx=i)
            outputs = self.forward(batch, True, i)
            opt.zero_grad()
            self.manual_backward(outputs['loss'])
            opt.step()
            self.log(loss_name, outputs['loss'].item(), prog_bar=True)
            if n_opt > 1:
                self.untoggle_optimizer(opt)

    def get_prediction_path(self, batch):
        epoch_pred_fstr = self.predictions_fstr % (str(self.current_epoch))
        variable_keys = re.findall(r'{(.*?)}', epoch_pred_fstr)
        # Get path variables
        fstr_var_values = []
        # Flatten a batch of multiple video frame_ids into a list of strings.
        for key_values in zip(*[batch[key] for key in variable_keys]):
            key_vals = [
                val.split(self.sep) if self.sep is not None else [val] for val in key_values]
            fstr_var_values += list(itertools.product(*key_vals)).copy()
        # Create save path for each frame.
        prediction_file_paths = []
        for vals in fstr_var_values:
            fvals = {k: v for k, v in zip(variable_keys, vals)}
            prediction_file_paths.append(epoch_pred_fstr.format(**fvals))
        return prediction_file_paths

    def to_contours(self, prob, input_image=None, **kwargs):
        prob = prob.clone().detach().cpu()
        masks = prob.cpu().numpy() > self.bg_conf_thresh
        color_segments = visual_utils.draw_contours(masks, input_image)
        return color_segments

    def to_label_colors(self, prob, cls_prob, **kwargs):
        """
        Transform a single image proposal masks and classes into color map.
        :param probs: (torch.tensor) n_proposals * H * W.
        :param cls_probs: (torch.tensor) n_proposals * n_classes.
        :return: (numpy.array) H * W * 3. A color image.
        """
        prob = prob.detach().cpu()
        cls_prob = cls_prob.detach().cpu()
        proposal_mask = prob > self.bg_conf_thresh
        max_cls_prob, cls_labels = cls_prob.max(-1)
        proposal_labels = (
            proposal_mask * cls_labels.unsqueeze(-1).unsqueeze(-1)).long()
        # Note: Here we assume the zero dimension of the class is always background class.
        # The not background class is the probability that the proposal mask contains an object.
        valid_proposal = 1 - cls_prob[:, 0] > self.conf_thresh
        if valid_proposal.sum() == 0:
            valid_proposal = (max_cls_prob == max_cls_prob.max())
        valid_probs = max_cls_prob[valid_proposal]
        proposal_labels = proposal_labels[valid_proposal]
        #proposal_mask = proposal_mask[valid_proposal]
        segment_labels = torch.zeros_like(proposal_labels[0])
        for i in valid_probs.argsort():
            segment_labels = torch.where(
                proposal_labels[i] != 0, proposal_labels[i], segment_labels)
            #segment_labels[proposal_mask[i]] = proposal_labels[i][proposal_mask[i]]
        segment_labels = segment_labels.numpy()
        # H * W * 3
        segment_label_colors = visual_utils.colorize_labels(segment_labels, self.label_colors)
        return segment_label_colors, max_cls_prob.max().item()

    def validation_step(self, batch, batch_nb):
        outputs = self(
            batch, compute_loss=True, optimizer_idx=self.val_optimizer_idx)
        loss = outputs['loss'].item()
        if (self.predictions_fstr is not None) and (self.max_val_viz_batch > batch_nb):
            predictions_paths = self.get_prediction_path(batch)
            fidx = sum([list(zip([i] * q.size(0), range(q.size(0)))) 
                    for i, q in enumerate(outputs['cls_prob'])], [])
            for i, frame in enumerate(ops.flatten_list(batch['frames'])):
                save_path = predictions_paths[i]
                iv, ind = fidx[i]
                q = outputs['prob'][iv][ind]
                q_cls = outputs['cls_prob'][iv][ind]
                segment_conts = self.to_contours(q, frame.detach().cpu().numpy())
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                PilImage.fromarray(segment_conts.astype('uint8')).save(save_path)
                segment_label_colors, max_prob = self.to_label_colors(q, q_cls)
                label_col_save_path = save_path.replace('.jpg', '_labcol%.4f.jpg' % max_prob)
                PilImage.fromarray(
                        segment_label_colors.astype('uint8')).save(label_col_save_path)
        self.log_dict({'val_loss': loss}, batch_size=self.bsize, on_step=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        return [torch.optim.AdamW(params=self.parameters(), lr=self.lr)], []


class BaseSegmentor(SegmentationEngine):
    def __init__(self, config, multi_fname_sep=None):
        super(BaseSegmentor, self).__init__(
            batch_size=config['batch_size'],
            predictions_fstr=config.get('val_prediction_fstr', None),
            predictions_fsep=multi_fname_sep,
            min_confidence=config.get('min_confidence', 0.3),
            bg_min_confidence=config.get('bg_min_confidence', 0.3),
        )
        self.backbone = None
        self.head = None
        self.config = config
        self.num_proposals = self.config.get('max_num_objects', 20)
        self.n_classes = self.config.get('num_classes', 1)
        self.build_network()
        self.val_optimizer_idx = 0
        self.nll = torch.nn.NLLLoss()
        self.multi_ops = ops.MultiSplitPairwiseOps(
            segment_ops=[ops.pairwise_cross_entropy, ops.pairwise_bce, ops.pairwise_mask_iou_dice],
            dimension_split_ids=[self.n_classes, self.n_classes + 1],
            weights=[0.1, 1.0, 1.0])
        # Focal loss shows some NaN loss values.
        self.label_colors = color_utils.generate_colors(self.n_classes)
        self.hung_max_iter = self.config.get('hungarian_max_iterations', 0)

    def build_network(self):
        #self.gen = pspnet.build_network(**self.config['pspnet'])
        #self.gen = HourglassNetSimple(Bottleneck, **self.config['hourglass'])
        #self.backbone = hourglass.HourglassNet(hourglass.Bottleneck, **self.config['hourglass'])
        #self.config['generator']['in_channels'] = self.config['hourglass']['num_classes']
        self.head = pix2pix.Generator(
            out_channels=self.num_proposals, 
            n_aux_classes=self.n_classes,
            **self.config['generator'])

    def configure_optimizers(self):
        b1, b2 = 0.5, 0.999
        opt = torch.optim.AdamW(
            self.parameters(), betas=(b1, b2), **self.config['optimizer'])
        self.loss_names = ['label_plus_obj_loss']
        return [opt], []

    def forward_resizes(self, batch, n_frames_sizes):
        xs = [f.permute(2, 0, 1) / 255.0 for f in ops.flatten_list(batch['frames'])]
        xs = [pspnet.resnet_preprocess(x) for x in xs]
        xs = torch.stack(xs)
        if self.backbone is not None:
            xs = self.backbone(xs)
        # (batch_size * n_proposal * H * W), (batch_size * n_proposal * n_classes)
        y_logits, y_cls = self.head(xs)
        output_logits = []
        output_cls_logits = []
        idx = 0
        for n, fsize in n_frames_sizes:
            logits = torch.nn.functional.interpolate(
                y_logits[idx: idx + n], size=fsize, mode='bilinear')
            output_logits.append(logits)
            output_cls_logits.append(y_cls[idx: idx + n])
            idx += n 
        return output_logits, output_cls_logits

    def forward_patches(self, batch, n_frames_sizes):
        xs = [f.permute(2, 0, 1) / 255.0 for f in ops.flatten_list(batch['frames'])]
        x_patches, x_coords = image_utils.image_tensors_to_patches(xs, self.config['patch_size'])
        x_patches = x_patches.to(device).float()
        if self.backbone is not None:
            x_patches = self.backbone(x_patches)
        y_logits = self.head(x_patches)
        y_logits = image_utils.patches_to_image_tensors(y_logits, x_coords)
        idx = 0
        output_logits = []
        for n, fsize in n_frames_sizes:
            logits = torch.stack(y_logits[idx: idx + n])
            output_logits.append(logits)
            idx += n 
        return  output_logits
    
    def custom_obj_loss(self, target_one_hot, q):
        """
        Custom object los for each frame. We will have to calculate hungarian loss for each frame
        because each video frame may have different number of true objects.

        :param target_one_hot: A pixel-wise annotated for a single video frame in shape n_frames *
            n_objs * image_width * image_height.
        :param q: A pixel-wise predicted object proposal tensor for a single video frame in shape
            n_frames * n_objs * image_width * image_height.
        :return: Average IoU dice loss/ hungarian loss per frame calculated based on the closest 
            matched object mask.
        """
        # Sigmoid + hungarian iou loss works.
        iou_loss = hungarian.hungarian_loss(target_one_hot, q, ops.pairwise_mask_iou_dice)
        return iou_loss.mean()

    def custom_annotation_loss(self, target_onehot, q_cls):
        """
        Custom annotation loss for each proposal object.

        :param target_one_hot: n_frames * n_proposals * n_classes. Onehot groundtruth object 
            annotation class labels.
        :param q_cls: n_frames * n_proposals *  n_classes. Predicted softmax class label
            probabilities for each proposal mask.
        :return: Average of Huangarian cross entropy per frame.
        """
        #annotation_loss = hungarian.hungarian_loss(target_onehot, q_cls, ops.pairwise_bce)
        annotation_loss = hungarian.hungarian_loss(target_onehot, q_cls, self.multi_ops)
        return annotation_loss.mean()

    def forward(self, batch, compute_loss=False, optimizer_idx=0):
        n_frames_sizes = [(len(fs), fs[0].size()[:2]) for fs in batch['frames']]
        output_logits, output_cls_logits = self.forward_resizes(batch, n_frames_sizes)
        # smoothed softmax: https://stackoverflow.com/questions/44081007/logsoftmax-stability
        #bs = [torch.max(logits, dim=1, keepdim=True).values for logits in output_cls_logits]
        #q_cls = [torch.softmax(logits - b, dim=1) for b, logits in zip(bs, output_cls_logits)]
        # n_frames * n_proposal * n_classes
        q_cls = [torch.softmax(logits, dim=-1) for logits in output_cls_logits]
        # n_frames * n_proposal * n_classes * H * W, Softmax along class dimensions.
        q = [torch.sigmoid(logits) for logits in output_logits]
        loss = None
        if compute_loss:
            batch_gt = data_utils.collate_dict(batch['gt_frames']) 
            loss = 0.0
            for iv in range(len(n_frames_sizes)):
                # The number of objects in each frame varies. We can't proposed background for
                # generality.
                gt_obj_masks = [pad_or_crop_obj_mask(mask, self.num_proposals)
                    for mask in batch_gt['obj_mask'][iv]]
                obj_mask_onehot = torch.stack(gt_obj_masks).float().contiguous()
                # This only applies on background label is assumed to be 0 dim.
                #loss += self.custom_obj_loss(obj_mask_onehot, q[iv])
                # n_frames * H * W
                label_masks = torch.stack(batch_gt['label_mask'][iv])
                # n_frames * n_proposal * H * W
                obj_label_masks = obj_mask_onehot * label_masks.unsqueeze(1)
                # This only applies when background label is assumed to be 0 dim.
                # n_frames * n_proposal * (HW)
                flatten_obj_label_masks = obj_label_masks.view(*obj_label_masks.size()[:-2], -1)
                obj_label_count = torch.count_nonzero(flatten_obj_label_masks, dim=-1)
                obj_label_count = torch.where(
                    obj_label_count == 0, torch.tensor(1e-6), obj_label_count)
                # n_frames * n_proposal 
                obj_labels = flatten_obj_label_masks.sum(-1) / obj_label_count
                # n_frames * n_proposal * n_classes
                obj_labels_onehot = torch.nn.functional.one_hot(
                    obj_labels.long(), num_classes=self.n_classes)
                obj_labels_onehot = obj_labels_onehot.float().contiguous()
                # n_frames * n_proposal. The onehot slice zero = 1 denotes background class.
                obj_onehot = 1 - ops.tensor_slice(obj_labels_onehot, -1, slice_id=0)
                # n_frames * n_proposal * 1
                obj_onehot = obj_onehot.unsqueeze(-1)
                # n_frames * n_proposal * HW
                flatten_obj_mask_onehot = obj_mask_onehot.view(*obj_mask_onehot.size()[:-2], -1)
                # n_frames * n_proposal * (HW + 1 + n_classes)
                ext_obj_labels_onehot = torch.concat(
                    [obj_labels_onehot, obj_onehot, flatten_obj_mask_onehot], dim=-1)
                # n_frames * n_proposal * HW
                flatten_q_iv = q[iv].view(*q[iv].size()[:-2], -1)
                q_obj = 1 - ops.tensor_slice(q_cls[iv], -1, slice_id=0)
                q_obj = q_obj.unsqueeze(-1)
                # n_frames * n_proposal * (HW + 1 + n_classes)
                ext_q_cls_iv = torch.concat([q_cls[iv], q_obj, flatten_q_iv], dim=-1)
                loss += self.custom_annotation_loss(ext_obj_labels_onehot, ext_q_cls_iv)
                #loss += self.custom_annotation_loss(obj_labels_onehot, q_cls[iv])
            loss /= len(n_frames_sizes)
        return {'prob': q, 'cls_prob': q_cls, 'loss': loss}

