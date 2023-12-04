
from utils import color_utils
from preprocessors.constants import *
from preprocessors.datasets.a2d import *
from preprocessors.datasets.bsds import *
from preprocessors.datasets.kth import *
from preprocessors.datasets.prnews import *
from preprocessors.datasets.ucf import *
from preprocessors.datasets.hmdb51 import *
from preprocessors.datasets.something_something_v2 import *


VID_TXT_COLS = [SAMPLE_ID_KEY, CLIP_PATH_KEY, SPLIT_KEY, CLASS_NAME, FRAME_ID_KEY]
VID_RECG_COLS = [SAMPLE_ID_KEY,
                 CLIP_PATH_KEY,
                 SPLIT_KEY,
                 CLASS_NAME,
                 CLASS_ID_KEY,
                 FRAME_ID_KEY]
VID_RECG_TRAIN_SPLIT_CSV = 'data_model/video_recognition_{split}.csv'
VID_RECG_TRAIN_CLSNAME_MAP = 'data_model/video_recognition_clsname.json'


def video_recognition(dataset_configs):
    dfs = []
    clsname_map = []
    for ds_name, config in dataset_configs.items():
        fn = globals().get(config['pipeline_fn'])
        df = fn(**config)
        if 'clsname_map' in df.columns:
            clsname_map.append(json.loads(df['clsname_map'][0]))
        cols = df.columns.intersection(VID_RECG_COLS)
        df = df[cols]
        df[DATASET_KEY] = ds_name
        for col in set(VID_RECG_COLS) - set(cols):
            df[col] = ''
        dfs.append(df)
    dfs = pd.concat(dfs)
    if clsname_map:
        consistent = all(str(x) == str(clsname_map[0]) for x in clsname_map)
        clsname_map = None if not consistent else clsname_map[0]
    with open(VID_RECG_TRAIN_CLSNAME_MAP, 'w') as f:
        if clsname_map is not None:
            json_data = clsname_map
        else:
            json_data = dfs[[CLASS_ID_KEY, CLASS_NAME]]
            json_data = json_data.set_index(CLASS_ID_KEY)[CLASS_NAME].to_dict()
        json.dump(json_data, f)
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df.to_csv(VID_RECG_TRAIN_SPLIT_CSV.format(split=split), index=False)


VIDTXT_TRAIN_SPLIT_CSV = 'data_model/video_text_mixed_{split}.csv'
VIDTXT_ALL_TEXTS = 'data_model/video_text_mixed_all_texts.txt'


def mixed_video_text(dataset_configs):
    dfs = []
    for ds_name, config in dataset_configs.items():
        fn = globals().get(config['pipeline_fn'])
        df = fn(**config)
        cols = df.columns.intersection(VID_TXT_COLS)
        df = df[cols]
        df[DATASET_KEY] = ds_name
        for col in set(VID_TXT_COLS) - set(cols):
            df[col] = ''
        dfs.append(df)
    dfs = pd.concat(dfs)
    all_texts = set()
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df[TEXT_KEY] = df[CLASS_NAME]
        df.to_csv(VIDTXT_TRAIN_SPLIT_CSV.format(split=split), index=False)
        all_texts.update(set(df[TEXT_KEY].dropna()))
    with open(VIDTXT_ALL_TEXTS, 'w') as f:
        for t in sorted(all_texts):
            f.write(t + '\n')


VID_SEG_TRAIN_SPLIT_CSV = 'data_model/video_segmentation_{split}.csv'
VID_SEG_COLS = [SAMPLE_ID_KEY, 
                CLIP_PATH_KEY, 
                SPLIT_KEY,
                ANNOTATED_FRAME_ID,
                ANNOTATED_FRAME_PATH,
                COLOR_CODE,
                OBJECT_MASK_KEY,
                REF_TEXT_KEY,
                LABEL_MASK_KEY,
                ANNOTATION_PROCESSOR]
VID_SEG_DATASET_SEP = '\t'


def mixed_video_segmentation(dataset_configs):
    dfs = []
    for ds_name, config in dataset_configs.items():
        fn = globals().get(config['pipeline_fn'])
        df = fn(**config)
        cols = df.columns.intersection(VID_SEG_COLS)
        df = df[cols]
        df[DATASET_KEY] = ds_name
        for col in set(VID_SEG_COLS) - set(cols):
            df[col] = ''
        assert COLOR_CODE in df.columns, \
            f'Video segmentation dataset {ds_name} error: Missing color code.'
        df_color_code = df[[DATASET_KEY, COLOR_CODE]]
        df_color_code = df_color_code.set_index(DATASET_KEY)[COLOR_CODE]
        uniq_color_codes = [json.loads(color_code) 
                            for color_code in df_color_code.unique().tolist()]
        uniq_color_id_map, _, _ = color_utils.uniq_color_id_coder(uniq_color_codes)
        df[UNIQUE_CLASS_ID_MAP] = json.dumps(uniq_color_id_map)
        dfs.append(df)
    dfs = pd.concat(dfs)
    for split in dfs[SPLIT_KEY].unique():
        df = dfs[dfs[SPLIT_KEY] == split]
        df.to_csv(VID_SEG_TRAIN_SPLIT_CSV.format(split=split), 
                  sep=VID_SEG_DATASET_SEP, 
                  index=False)