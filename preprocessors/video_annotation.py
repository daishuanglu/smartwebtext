import numpy as np
import h5py
from collections import namedtuple

from preprocessors.datasets import a2d

AnnotationContext = namedtuple(
    'AnnotationContext', ['label_mask', 'obj_mask', 'ref_text'])


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


class FrameAnnotation:

    def __init__(self, 
                 label_mask_key,
                 object_mask_key,
                 processor,
                 ref_text_key=None,
                 unique_cls_id_map={},
                 color_code={},
                 **kwargs) -> None:
        self.obj_mask_key = object_mask_key
        self.cls_id_map = lambda x: unique_cls_id_map.get(x, x)
        original_cls_id = {v: k for k, v in unique_cls_id_map.items()}
        self.original_cls_id_map = lambda x: original_cls_id.get(x, x)
        self.color_code = color_code
        self.label_mask_key = label_mask_key
        self.ref_text_key = ref_text_key
        self.processor_fn = processor

    def __call__(self, annotation_path, **kwargs):
        context = globals()[self.processor_fn](
            annotation_path,
            self.label_mask_key,
            self.ref_text_key,
            self.obj_mask_key,
            **kwargs)
        label_mask = np.vectorize(self.cls_id_map)(context.label_mask)
        self.context = AnnotationContext(label_mask, context.obj_mask, context.ref_text)

    def from_prediction(self, pred_path) -> AnnotationContext:
        # TODO: Function to transform predicted color image/ id map to AnnotationContext.
        return
    

if __name__ == '__main__':
    test_anno_path = 'D:/video_datasets/a2d_annotation_with_instances/_0djE279Srg/00015.h5'
    annotation = FrameAnnotation(object_mask_key='reMask', 
                                 label_mask_key='reS_id', 
                                 ref_text_key='instance')
    df = a2d.load_ref_text_df(root='D:/video_datasets')
    annotation(test_anno_path, 'a2d_context', ref_text_df=None)
    print(annotation.context)