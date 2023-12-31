import numpy as np
import h5py
from collections import namedtuple
import torch
import pickle

from preprocessors.datasets import a2d
from preprocessors.pipelines import (
  CLIP_PATH_KEY,
  ANNOTATION_PROCESSOR,
  OBJECT_MASK_KEY,
  LABEL_MASK_KEY,
  REF_TEXT_KEY,
  FRAME_ID_SEP, 
  SAMPLE_ID_KEY,
  ANNOTATED_FRAME_ID,
  ANNOTATED_FRAME_PATH,
  DATASET_KEY,
  UNIQUE_CLS_ID_PATH
)


AnnotationContext = namedtuple(
    'AnnotationContext', ['label_mask', 'obj_mask', 'ref_text'])


class VideoMetadata():

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def unique_class_id_map(split=''):
        with open(UNIQUE_CLS_ID_PATH.format(split=split), 'rb') as fp:
            unique_cls_id_map = pickle.load(fp)
        return unique_cls_id_map
    
    @staticmethod
    def dataset_name(meta_dict):
        return meta_dict.get(DATASET_KEY, None)

    @staticmethod
    def clip_path_key(meta_dict):
        return meta_dict.get(CLIP_PATH_KEY, None)
    
    @staticmethod
    def label_mask_key(meta_dict):
        return meta_dict.get(LABEL_MASK_KEY, None)
    
    @staticmethod
    def object_mask_key(meta_dict):
        return meta_dict.get(OBJECT_MASK_KEY, None)
    
    @staticmethod
    def ref_text_key(meta_dict):
        return meta_dict.get(REF_TEXT_KEY, None)
    
    @staticmethod
    def frame_ids(meta_dict):
        return meta_dict.get(ANNOTATED_FRAME_ID, '').split(FRAME_ID_SEP)
    
    @staticmethod
    def video_id(meta_dict):
        return meta_dict.get(SAMPLE_ID_KEY, None)
    
    @staticmethod
    def processor_fn_name(meta_dict):
        return meta_dict.get(ANNOTATION_PROCESSOR, None)

    @staticmethod
    def frame_annotation_paths(meta_dict):
        return meta_dict.get(ANNOTATED_FRAME_PATH, '').split(FRAME_ID_SEP)

    @staticmethod
    def save_sequence_predictions(pred_dict, frame_ids, output_path, bg_conf, min_conf):
        with h5py.File(output_path, 'w') as f:
            for key, values in pred_dict.items():
                if isinstance(values, torch.Tensor):
                    # Convert to NumPy array
                    values = values.cpu().detach()
                    values = np.round(values, decimals=5)
                    f.create_dataset(key, data=values, compression='gzip')
            f.create_dataset('frame_ids', data=frame_ids)
            f.create_dataset('bg_conf', data=bg_conf)
            f.create_dataset('min_conf', data=min_conf)

    @staticmethod
    def load_sequence_predictions(pred_file_path):
        with h5py.File(pred_file_path, 'r') as f:
            result = {
                'prob': np.array(f['prob'][:]),
                'cls_prob': np.array(f['cls_prob'][:]),
                'frame_ids': f['frame_ids'],
                'bg_conf': float(f['bg_conf']),
                'min_conf': float(f['min_conf'])
                }
            return result
        
    @staticmethod
    def frame_annotations(meta_dict, unique_cls_id_map={}, **kwargs):
        contexts = []
        cls_id_map = lambda x: unique_cls_id_map.get(x, None)
        annotation_paths = VideoMetadata.frame_annotation_paths(meta_dict)
        annotation_paths = [p for p in annotation_paths if p]
        for annotation_path in annotation_paths:
            context = globals()[VideoMetadata.processor_fn_name(meta_dict)](
                annotation_path,
                VideoMetadata.label_mask_key(meta_dict),
                VideoMetadata.ref_text_key(meta_dict),
                VideoMetadata.object_mask_key(meta_dict),
                **kwargs)
            label_mask = np.vectorize(cls_id_map)(context.label_mask)
            contexts.append(AnnotationContext(label_mask, context.obj_mask, context.ref_text))
        return contexts

    @staticmethod
    def prediction_to_masks(prob, cls_prob, min_conf, bg_conf):
        proposal_mask = prob > bg_conf
        max_cls_prob, cls_labels = cls_prob.max(-1), cls_prob.argmax(-1)
        proposal_labels = (
            proposal_mask * cls_labels.unsqueeze(-1).unsqueeze(-1)).long()
        # Note: Here we assume the zero dimension of the class is always background class.
        # The not background class is the probability that the proposal mask contains an object.
        valid_proposal = 1 - cls_prob[:, 0] > min_conf
        if valid_proposal.sum() == 0:
            valid_proposal = (max_cls_prob == max_cls_prob.max())
        valid_probs = max_cls_prob[valid_proposal]
        proposal_labels = proposal_labels[valid_proposal]
        segment_labels = np.zeros_like(proposal_labels[0])
        for i in valid_probs.argsort():
            segment_labels = np.where(
                proposal_labels[i] != 0, proposal_labels[i], segment_labels)
        return segment_labels, proposal_mask[valid_proposal]
    

def a2d_context(annotation_path, 
                label_mask_key,
                ref_text_key,
                obj_mask_key,
                ref_text_df=None,
                **kwargs) -> AnnotationContext:
    vid = a2d.vid_from_annotation_path(annotation_path)
    ds_root = a2d.root_from_annotation_path(annotation_path)
    with h5py.File(annotation_path, 'r') as mat_file:
        mat_obj = np.array(mat_file[obj_mask_key][...])
        if len(mat_obj.shape) == 2:
            mat_obj = mat_obj[np.newaxis, :, :]  # (num_obj, img_height, img_width)
        mat_obj = np.array(mat_obj).transpose(0, 2, 1)
        ref_text = []
        if ref_text_key:
            for obj_id in np.array(mat_file[ref_text_key], dtype=int):
                text = a2d.get_ref_text(root=ds_root, 
                                        vid=vid, 
                                        obj_id=obj_id, 
                                        df=ref_text_df)
                ref_text.append(text)
        # (img_height, img_width)
        mat_label = np.array(mat_file[label_mask_key][...]).T
    return AnnotationContext(mat_label, mat_obj, ref_text)

