import math
import json
import random
from collections import namedtuple, defaultdict
from typing import List, Dict
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


PLOT_MARKERS = lambda x: list(Line2D.markers)[x % len(Line2D.markers)]
PLOT_CSS4_COLORS = lambda x: list(mcolors.CSS4_COLORS)[x % len(mcolors.CSS4_COLORS)]
PLOT_TAB_COLORS = lambda x:  list(mcolors.TABLEAU_COLORS)[x % len(mcolors.TABLEAU_COLORS)]
PLOT_BASE_COLORS = lambda x: list(mcolors.BASE_COLORS)[x % len(mcolors.BASE_COLORS)]
ColorCode = namedtuple('ColorCode', ['id', 'name', 'color'])
MSCOCO_NUM_INSTANCE = 80
# Classes
MSCOCO_OBJ_NAMES ={
  0: 'person',
  1: 'bicycle',
  2: 'car',
  3: 'motorcycle',
  4: 'airplane',
  5: 'bus',
  6: 'train',
  7: 'truck',
  8: 'boat',
  9: 'traffic light',
  10: 'fire hydrant',
  11: 'stop sign',
  12: 'parking meter',
  13: 'bench',
  14: 'bird',
  15: 'cat',
  16: 'dog',
  17: 'horse',
  18: 'sheep',
  19: 'cow',
  20: 'elephant',
  21: 'bear',
  22: 'zebra',
  23: 'giraffe',
  24: 'backpack',
  25: 'umbrella',
  26: 'handbag',
  27: 'tie',
  28: 'suitcase',
  29: 'frisbee',
  30: 'skis',
  31: 'snowboard',
  32: 'sports ball',
  33: 'kite',
  34: 'baseball bat',
  35: 'baseball glove',
  36: 'skateboard',
  37: 'surfboard',
  38: 'tennis racket',
  39: 'bottle',
  40: 'wine glass',
  41: 'cup',
  42: 'fork',
  43: 'knife',
  44: 'spoon',
  45: 'bowl',
  46: 'banana',
  47: 'apple',
  48: 'sandwich',
  49: 'orange',
  50: 'broccoli',
  51: 'carrot',
  52: 'hot dog',
  53: 'pizza',
  54: 'donut',
  55: 'cake',
  56: 'chair',
  57: 'couch',
  58: 'potted plant',
  59: 'bed',
  60: 'dining table',
  61: 'toilet',
  62: 'tv',
  63: 'laptop',
  64: 'mouse',
  65: 'remote',
  66: 'keyboard',
  67: 'cell phone',
  68: 'microwave',
  69: 'oven',
  70: 'toaster',
  71: 'sink',
  72: 'refrigerator',
  73: 'book',
  74: 'clock',
  75: 'vase',
  76: 'scissors',
  77: 'teddy bear',
  78: 'hair drier',
  79: 'toothbrush'
}


def uniq_color_code_map(color_codes: List[List[Dict]]):
    uniq_color_code = []
    uniq_color_names = []
    uniq_ids = []
    segment_ids = []
    i = 0
    for seg_id, color_code in enumerate(color_codes):
        for code in color_code:
            name = code['name'].lower()
            if name in uniq_color_names:
                uniq_ids.append(uniq_color_names.index(name))
            else:
                uniq_color_code.append(code)
                uniq_color_names.append(name)
                uniq_ids.append(i)
                segment_ids.append(seg_id)
                i += 1
    return uniq_color_code, uniq_ids, segment_ids


def uniq_color_code(ds_color_codes: List[List[ColorCode]]):
    uniq_color_names = []
    uniq_ids = []
    color_codes = [(color_code, ds_id) 
                   for ds_id, ds_color_code in enumerate(ds_color_codes) 
     for color_code in ds_color_code]
    ds_cc_ids = []
    ds_ids = []
    for color_code, ds_id in color_codes:
        if color_code.name not in uniq_color_names:
            uniq_ids.append(len(uniq_color_names))
            uniq_color_names.append(color_code.name)
        else:
            uniq_ids.append(uniq_color_names.index(color_code.name))
        ds_cc_ids.append(color_code.id)
        ds_ids.append(ds_id)
    uniq_colors = generate_colors(len(uniq_color_names))
    uniq_color_code = [ColorCode(ic, c[0], c[1]) for ic, c in 
                       enumerate(zip(uniq_color_names, uniq_colors))]
    uniq_ids_map = defaultdict(dict)
    for i, ds_id in enumerate(ds_ids):
        uniq_ids_map[ds_id].update({ds_cc_ids[i]: uniq_ids[i]})
    return uniq_color_code, uniq_ids_map


def save_color_codes(color_codes: List[ColorCode], fpath: str):
    dicts = []
    for cc in color_codes:
        dicts.append(cc._asdict())
    with open(fpath, 'w') as f:
        json.dump(dicts, f)


def load_color_codes(fpath: str) -> List[ColorCode]:
    with open(fpath, 'r') as f:
        color_codes = [ColorCode(**d) for d in json.load(f)]
    return color_codes


def generate_colors(num_labels, shuffle=True):
    # Compute the number of different R, G, B values
    m = int(math.ceil(num_labels ** (1/3)))
    p = int(math.ceil(num_labels / m) ** 0.5)
    q = int(math.ceil(num_labels / (m * p)))

    # Compute the step size for each color channel
    step_size = int(256 / max(m, p, q))

    # Create a list of RGB colors
    colors = []
    for i in range(m):
        for j in range(p):
            for k in range(q):
                # Compute the current RGB color by multiplying the step size by the index
                color = (i * step_size, j * step_size, k * step_size)
                colors.append(color)
    bg = colors.pop(0)
    colors = colors[:num_labels]
    if shuffle:
        random.shuffle(colors)
    return [bg] + colors


if __name__ == '__main__':
    colors = generate_colors(128)
    print(len(colors))
    for c in colors:
        print(c)