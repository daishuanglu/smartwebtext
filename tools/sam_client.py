"""
Facebook segment anything model
https://github.com/facebookresearch/segment-anything
Demo website: https://segment-anything.com/demo#

pip install git+https://github.com/facebookresearch/segment-anything.git
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
"""

import os
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import pims
from tqdm import tqdm
from utils import download_utils
from utils.train_utils import device
from utils import color_utils

VIT_H_SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
VIT_L_SAM_URL  = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
VIT_B_SAM_URL  = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
MAX_NUM_COLORS = 500
# remove the (0,0,0) background color
SEGMENT_COLORS = color_utils.generate_colors(MAX_NUM_COLORS+1)[1:]


class SAMClient():

    model_urls = {
        'vit_b':  VIT_B_SAM_URL,
        'vit_l': VIT_L_SAM_URL,
        'vit_h': VIT_H_SAM_URL
    }

    def __init__(self, model_dir, model_type=None):
        model_type = "vit_b" if model_type is None else model_type
        model_url = self.model_urls[model_type]
        model_name = os.path.basename(model_url)
        model_path = os.path.join(model_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(model_path):
            download_utils.download_url(model_url, model_path)
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def segmented(self, image: np.array):
        masks = self.mask_generator.generate(image)
        colors = SEGMENT_COLORS[:len(masks)].copy()
        color_mask = np.zeros_like(image)
        for mask, color in zip(masks, colors):
            color_mask[mask['segmentation']] = color
        return color_mask

    def segment_video(self, video_path, output_fname):
        os.makedirs(os.path.dirname(output_fname), exist_ok=True)
        v = pims.Video(video_path)
        #print('processing video clip %s' % video_path)
        # Define the output video codec using fourCC code (H.264 codec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width, height = v.frame_shape[1], v.frame_shape[0]
        writer = cv2.VideoWriter(output_fname, fourcc, 30.0, (width, height))
        for i in tqdm(range(len(v)), desc=os.path.basename(video_path)):
            segments = self.segmented(np.array(v[i]))
            #print(segments)
            segments = segments.astype(np.uint8)
            writer.write(segments)
        writer.release()


if __name__=='__main__':
    HOME = '/home/shuangludai'
    test_video_path = '%s/A2D/A2D_main_1_0/Release/clips320H/_KIlkrVZb-g.mp4' % HOME
    test_segmented_video_path = '%s/A2D/A2D_main_1_0/Release/SAMclips320H/_KIlkrVZb-g.mp4' % HOME
    sam_model_dir = '%s/ai_models/segment_anything' % HOME
    sam_client = SAMClient(model_dir=sam_model_dir)
    sam_client.segment_video(test_video_path, test_segmented_video_path)
