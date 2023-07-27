import av
import numpy as np
import gzip
import PIL.Image as PilImage
from transformers import AutoProcessor, \
    AutoImageProcessor, \
    Mask2FormerForUniversalSegmentation, \
    AutoModelForCausalLM

import torch
import cv2
from utils import color_utils
import pims
from preprocessors import pipelines
from munkres import Munkres, print_matrix, DISALLOWED
from scipy.ndimage import measurements
from sklearn.metrics import euclidean_distances
from utils import train_utils
import pandas as pd
from tqdm import tqdm


git_vid_processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
git_vid_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")
np.random.seed(45)
# load Mask2Former fine-tuned on COCO instance segmentation
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-coco-instance").to(train_utils.device)
#git_processor = AutoProcessor.from_pretrained("microsoft/git-base")
#git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
#device = "cuda" if torch.cuda.is_available() else "cpu"


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def vid_caption(file_path):
    container = av.open(file_path)
    num_frames = git_vid_model.config.num_image_with_embedding
    indices = sample_frame_indices(
        clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
    )
    frames = read_video_pyav(container, indices)
    pixel_values = git_vid_processor(images=list(frames), return_tensors="pt").pixel_values
    generated_ids = git_vid_model.generate(pixel_values=pixel_values, max_length=50)
    generated_text = git_vid_processor.batch_decode(generated_ids, skip_special_tokens=True)
    print("Generated caption:", generated_text)
    return generated_text


def find_center_of_binary_image(binary_image):
    # Calculate image moments
    moments = measurements.center_of_mass(binary_image)
    centroid_x, centroid_y = moments

    return centroid_x, centroid_y


def munkres_matching(cost_matrix, verbose=False):
    m = Munkres()
    h, w = cost_matrix.shape
    l = h if h > w else w
    sq_cost_matrix = np.zeros((l, l))
    sq_cost_matrix[:h, :w] = cost_matrix.copy()
    #sq_cost_matrix = m.pad_matrix(cost_matrix.copy())
    indexes = m.compute(sq_cost_matrix)
    if verbose:
        print(cost_matrix.round(6))
        print_matrix(sq_cost_matrix, msg='Lowest cost through this matrix:')
        total = 0
        for row, column in indexes:
            value = sq_cost_matrix[row][column]
            total += value
            print(f'({row}, {column}) -> {value}')
        print(f'total cost: {total}')
    return indexes


def gzip_classifier(training_set, test_set, k):
    predictions = []
    for (x1 , _ ) in test_set:
        Cx1 = len(gzip.compress(x1.encode()))
        distance_from_x1 = []
        for (x2 , _ ) in training_set:
            Cx2 = len(gzip.compress(x2.encode()))
            x1x2 = " ". join ([x1 , x2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        sorted_idx = np.argsort (np.array(distance_from_x1 ) )
        top_k_class = training_set [sorted_idx[: k], 1]
        predict_class = max(set(top_k_class))
        predictions.append(predict_class)
    return predictions


def draw_segmentation_id(segmentation, segments_info):
    colors = color_utils.generate_colors(color_utils.MSCOCO_NUM_INSTANCE + 1)
    height, width = segmentation.shape
    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    for seg_info in segments_info:
        output_img[segmentation == seg_info['id']] = colors[seg_info['id'] + 1]
    return output_img


def test_segmentation():
    HOME = 'D:/backup_project/BSR_bsds500'
    test_image_path = '%s/BSDS500/data/images/train/8049.jpg' % HOME
    test_segmented_image_path = '%s/exmperimental/8049.jpg' % HOME
    image = PilImage.open(test_image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    segmented_color_map = draw_segmentation_id(**result)
    cv2.imwrite(test_segmented_image_path, segmented_color_map)
    return result


def instance_segmentation(np_img):
    pil_img = PilImage.fromarray(np_img)
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(outputs, target_sizes=[pil_img.size[::-1]])[0]
    return result


def find_isolated_area_bounding_boxes(binary_image):
    # Find contours in the binary image
    binary_image = np.asarray(binary_image, dtype=np.uint8)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate bounding boxes for all the contours
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    return bounding_boxes


def binary_image_to_string(binary_image):
    # Map 0 to '0' and 1 to '1', then join the rows to form a string
    binary_string = '\n'.join([''.join(['0' if pixel == 0 else '1' for pixel in row]) for row in binary_image])

    return binary_string


def instance_string(segmentation, segments_info):
    updated_info = []
    while segments_info:
        info = segments_info.pop(0)
        segment = segmentation.numpy().astype(int)
        region = info['id'] == segment
        #area = np.prod(np.array(region).shape) if region.any() else 0
        area = region.sum()
        if area < 100:
            print('Region too small for label id=%d, MSCOCO object_name=%s' % (
                info['id'], color_utils.MSCOCO_OBJ_NAMES[info['id']]))
            continue
        cnts = find_isolated_area_bounding_boxes(region)
        bbox = cnts[0]
        info['bbox'] = bbox
        x, y, w, h = bbox
        info['area'] = area ** 0.5
        repr_x = binary_image_to_string(segment[y:y+h, x:x+w] == info['id'])
        info['repr_x'] = repr_x
        #repr_y = binary_image_to_string(segment[y:y + h, x:x + w].transpose() == info['id'])
        #info['repr_y'] = repr_y
        #center_x, center_y = find_center_of_binary_image(segment[y:y+h, x:x+w])
        #info['center'] = np.array([center_x, center_y])
        if repr == '':
            print(segment[x:x+w, y:y+h])
        updated_info.append(info)
    return updated_info


def gzip_dist(str_code_set2, str_code_set1):
    ncd_cost = []
    for x1 in str_code_set1:
        Cx1 = len(gzip.compress(x1.encode()))
        distance_from_x1 = []
        for x2 in str_code_set2:
            Cx2 = len(gzip.compress(x2.encode()))
            x1x2 = " ". join ([x1 , x2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        ncd_cost.append(distance_from_x1)
    return np.array(ncd_cost)


def l2_dist(center_set2, center_set1):
    x2 = np.stack(center_set2)
    x1 = np.stack(center_set1)
    d = euclidean_distances(x1, x2)
    return d


def test_tracking():
    A2D_DIR = 'D:/video_datasets/A2D'
    vid = '__KkKB4wzrY'
    vid = '_0djE279Srg'
    vid = '_1UY2k5Mu3o'
    vid = '_1MUXIam4lA'
    vid = '_4JOpIKy0rA'
    vf = pipelines.A2D_CLIP_PATH.format(root=A2D_DIR, vid=vid)
    v = pims.Video(vf)
    prev_result_info = {}
    for i, frame in enumerate(v):
        # if i < 12:
        #    continue
        print('-' * 10, 'frame %d' % i, '-' * 10)
        result = instance_segmentation(frame)
        result_info = instance_string(**result)
        if prev_result_info:
            print([color_utils.MSCOCO_OBJ_NAMES[info['id']] for info in result_info])
            cost_matrix_x = gzip_dist([info['repr_x'] for info in result_info],
                                      [info['repr_x'] for info in prev_result_info])
            # cost_matrix_y = gzip_dist([info['repr_y'] for info in result_info],
            #                          [info['repr_y'] for info in prev_result_info])
            label_cost_matrix = np.array([[
                (info['label_id'] != prev_info['label_id']) * 100 for info in result_info]
                for prev_info in prev_result_info])
            # cost_matrix_c = l2_dist([info['center'] for info in result_info],
            #                        [info['center'] for info in prev_result_info])
            area_cost_matrix = np.array([[
                abs(info['area'] - prev_info['area']) for info in result_info]
                for prev_info in prev_result_info])
            # cost_matrix[label_cost_matrix] = DISALLOWED
            # matching_cost = cost_matrix.tolist()
            matching_cost = cost_matrix_x + label_cost_matrix + area_cost_matrix
            id_maps = munkres_matching(matching_cost, verbose=True)
            for prev_i, cur_i in id_maps:
                if cur_i < len(result_info) and (prev_i < len(prev_result_info)):
                    result_info[cur_i]['id'] = prev_result_info[prev_i]['id']
            print('previous bboxes=', [info['bbox'] for info in prev_result_info])
        track_obj_color_viz = draw_segmentation_id(result['segmentation'], result_info)
        cv2.imshow('test tracking %s' % (vid), track_obj_color_viz)
        prev_result_info = result_info.copy()
        # print('tracked objects=', [
        #    color_utils.MSCOCO_OBJ_NAMES[info['label_id']] for info in prev_result_info])
        cv2.waitKey(10)
        # cv2.destroyAllWindows()
    return

def load_kth_vid_caps_split(split):
    df_split = pd.read_csv(
        pipelines.KTH_SPLIT_CSV.format(split=split),
        dtype=str, keep_default_na=False, na_values=[], parse_dates=False)
    classes = []
    caps = []
    for i, row in tqdm(list(df_split.iterrows()), total=len(df_split)):
        text = vid_caption(row['video'])
        caps.append(text[0])
        classes.append(row['action'])
        print('action class=', row['action'])
    training_set = list(zip(caps, classes))
    return training_set


if __name__ == '__main__':
    KTH_DATASET_DIR = 'C:/Users/shud0/KTHactions'
    pipelines.kth_action_video_nobbox(KTH_DATASET_DIR)
    training_set = load_kth_vid_caps_split('train')
    validation_set = load_kth_vid_caps_split('val')
    training_set = training_set + validation_set
    test_set = load_kth_vid_caps_split('eval')
    test_predictions = gzip_classifier(training_set, test_set, k=5)
    n_correct = 0
    for gt, pred in zip(test_set, test_predictions):
        n_correct += (gt[1] == pred)
    print('KTH accuracy =', n_correct/ len(test_set))