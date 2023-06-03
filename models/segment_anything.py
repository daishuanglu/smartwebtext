import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np

from utils import download_utils
from utils.train_utils import device

VIT_H_SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
VIT_L_SAM_URL  = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
VIT_B_SAM_URL  = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'

#mask_generator_2 = SamAutomaticMaskGenerator(
#    model=sam,
#    points_per_side=32,
#    pred_iou_thresh=0.86,
#    stability_score_thresh=0.92,
#    crop_n_layers=1,
#    crop_n_points_downscale_factor=2,
#    min_mask_region_area=100,  # Requires open-cv to run post-processing
#)

class SAMClient():

    model_urls = {
        'vit_b':  VIT_B_SAM_URL,
        'vit_l': VIT_L_SAM_URL,
        'vit_h': VIT_H_SAM_URL
    }

    def __init__(self, model_dir, model_type=None, mask_generator_configs=[]):
        model_type = "vit_b" if model_type is None else model_type
        model_url = self.model_urls[model_type]
        model_name = os.path.basename(model_url)
        model_path = os.path.join(model_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(model_path):
            download_utils.download_url(model_url, model_path)
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        self.mask_generators =[SamAutomaticMaskGenerator(sam)]
        for config in mask_generator_configs:
            self.mask_generators.append(SamAutomaticMaskGenerator(sam, **config))

    def masks(self, image: np.array):
        masks = []
        for generator in self.mask_generators:
            masks.append(generator.generate(image))
        return masks