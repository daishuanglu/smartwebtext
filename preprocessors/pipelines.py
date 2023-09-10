import numpy as np
import os
from typing import Dict, List
import pandas as pd
import glob
from tqdm import tqdm
from utils import string_utils
from utils import color_utils
from preprocessors import dataloader


PRNEWS_FILEPATTERN = 'data_model/scrapped_news/webtext_thread_*.txt'
PRNEWS_SEPARATOR = '\t'
PRNEWS_HEADERS = ['Company', 'Stock', 'date', 'Title', 'Body']
PRNEWS_INVALID_KEYTERMS = ['follow us', 'twitter|', 'linkedin|', '.copyright', 'www.',
    'facebook:', 'visit:', 'twitter:', 'for more information', 'click here', 'email', 'phone', 'logo']
PRNEWS_SENTENCE_MIN_WORDS = 5
PRNEWS_PARAGRAPH_SEP = ';;;;'
PRNEWS_DATA_SEP = '\t'
PRNEWS_MUST_CONTAIN_COLS = ['Text', 'Company']
PRNEWS_EVAL_DIR = 'evaluation/prnews_accounting'

KTH_ACTION_SRC_CSV = 'data_model/kth_actions.csv'
KTH_ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
KTH_ACTION_DATA_CSV_SEP = ','
KTH_SPLIT_COL = 'split'
KTH_SPLIT_CSV = 'data_model/kth_actions_{split}.csv'
KTH_FRAME_FEATURE_SEP = ';'
#KTH_ACTION_HOME = '/home/shuangludai/KTHactions'
KTH_VIDEO_FILE = '{action}/person{pid}_{action}_{var}_uncomp.avi'

FASTTEXT_HOME = 'fasttext'


def prnews_text_preproc(s):
    #stopwords = string_utils.load_stopwords()
    s = s.lower()
    s = [w.strip() for w in s.split(PRNEWS_PARAGRAPH_SEP)]
    s = sum([string_utils.split_to_sentences(ss) for ss in s], [])
    for term in PRNEWS_INVALID_KEYTERMS:
        s = list(filter(lambda x: term not in x, s))
    s = [string_utils.remove_punct(sp) for sp in s]
    s = list(filter(lambda x: len(x.split()) > PRNEWS_SENTENCE_MIN_WORDS, s))
    s = [string_utils.clean_char(sp).strip() for sp in s]
    #s = [string_utils.remove_stopwords(sp, stopwords, sub=' ').strip() for sp in s]
    s = [string_utils.replace_consecutive_spaces(sp) for sp in s]
    return PRNEWS_PARAGRAPH_SEP.join(s).strip()


def prnews(
        output_files, split_ratio, vocabs: Dict[str, List[str]] = {}):

    FASTTEXT_TOOL = string_utils.fasttext_toolkits(fasttext_model_home=FASTTEXT_HOME)

    def _body(s):
        if not FASTTEXT_TOOL.detEN([s.lower()])[0]:
            return ''
        return prnews_text_preproc(s)

    def _title(s):
        s = string_utils.get_title_name(s, sep='-').strip()
        s = s if FASTTEXT_TOOL.detEN([s])[0] else ''
        return s

    def _company(s):
        return string_utils.getcompanyname(s, sep='-').strip()

    def _text(row):
        t, b = row['Title'].lower(), row['Body'].lower()
        if t == '' and b == '':
            return ''
        else:
            s = PRNEWS_PARAGRAPH_SEP.join(
                [row['Title'].lower(), row['Body'].lower()]).strip()
            company_name = string_utils.remove_company_suffix(row['Company'].lower()).strip()
            return s.replace(company_name, '')

    textfile = dataloader.BaseTextFile(
        fpath=PRNEWS_FILEPATTERN, sep=PRNEWS_SEPARATOR,
        types={'Body': _body, 'Company': _company, 'Title': _title},
        col_fns=[('Text', _text)], must_contain=PRNEWS_MUST_CONTAIN_COLS)
    textfile.write(
        output_files=output_files,
        split_ratio=split_ratio,
        cols=['Company', 'Title', 'Text'],
        sep=PRNEWS_DATA_SEP)
    for vocab_path, vocab_cols in vocabs.items():
        textfile.vocab(vocab_cols, vocab_path)


def prnews_concept_examples(fpath, text_col, ref_col, sep):
    df = pd.read_csv(
        fpath, sep=sep, dtype=str, parse_dates=False, na_values=[], keep_default_na=False)

    def _clean_row(r):
        s = prnews_text_preproc(str(r[text_col]).lower())
        c = string_utils.remove_company_suffix(str(r[ref_col]).lower())
        s = s.replace(c, '')
        return s

    paragraphs = df.apply(_clean_row, axis=1).tolist()
    ss = []
    for sentences in paragraphs:
        ss += sentences.strip().split(PRNEWS_PARAGRAPH_SEP)
    return ss


BSDS_SAMPLES = '{root}/images/{split}/*.jpg'
BSDS_GT_FILE = '{root}/groundTruth/{split}/{iid}.mat'
BSDS_SPLIT_CSV = 'data_model/bsds_{split}.csv'


def bsds500(dataset_dir):
    df = dict(image=[], gt=[], iid=[], split=[])
    for split in ['train', 'val', 'test']:
        data_paths = glob.glob(BSDS_SAMPLES.format(root=dataset_dir, split=split))
        for fpath in tqdm(data_paths, desc='bsds %s' % split):
            iid = os.path.basename(fpath).split('.')[0]
            gt_fpath = BSDS_GT_FILE.format(root=dataset_dir, split=split, iid=iid)
            df['image'].append(fpath)
            df['gt'].append(gt_fpath)
            df['iid'].append(iid)
            df['split'].append(split)
    df = pd.DataFrame(df)
    for split in df['split'].unique():
        df_split = df[df['split'] == split]
        df_split.drop(columns=['split'])
        df_split.to_csv(BSDS_SPLIT_CSV.format(split=split), index=False)
    return


A2D_METADATA_COLUMNS = [
    'vid', 'actor_action_label',
    'start_time', 'end_time', 'height', 'width', 'num_frames', 'num_labeled_frames', 'split_no']
A2D_METADATA_PATH = '{root}/A2D_main_1_0/Release/videoset.csv'
A2D_CLIP_PATH = '{root}/A2D_main_1_0/Release/clips320H/{vid}.mp4'
A2D_SAM_OUTPUT_CLIP_PATH = '{root}/A2D_main_1_0/Release/clips320H_sam/{vid}.mp4'
A2D_ANNOTATION_PATH = '{root}/A2D_main_1_0/Release/Annotations/mat/{vid}/{fid}.mat'
A2D_ANNO_COLOR_MAP_PATH ='{root}/A2D_main_1_0/Release/Annotations/col/{vid}/{fid}.png'
A2D_IMAGE_SPLIT_CSV = 'data_model/A2D_video_image_{split}.csv'
A2D_FID_SEP = ';;;'
A2D_INFO_PATH = '{root}/A2D_main_1_0/Release/README'


def a2d_annotated_frame_ids(root, vid):
    fids = []
    for anno_path in glob.glob(A2D_ANNOTATION_PATH.format(root=root, vid=vid, fid='*')):
        fid = os.path.basename(anno_path).replace('.mat', '')
        fids.append(fid)
    return A2D_FID_SEP.join(fids)


def a2d_video_images(dataset_dir, label_colors_json, train_val_ratio=[0.95, 0.05]):
    df_meta = pd.read_csv(
        A2D_METADATA_PATH.format(root=dataset_dir), header=None,
        dtype=str, na_values=[], parse_dates=False, keep_default_na=False)
    os.makedirs(os.path.dirname(A2D_SAM_OUTPUT_CLIP_PATH), exist_ok=True)
    df_meta.columns = A2D_METADATA_COLUMNS
    df_meta['split'] = df_meta['split_no'].apply(lambda x: np.random.choice(
        ['train', 'val'], 1, p=train_val_ratio, replace=True)[0] if x == '0' else 'test')
    df_meta['fids'] = df_meta['vid'].apply(lambda x: a2d_annotated_frame_ids(dataset_dir, x))
    with open(A2D_INFO_PATH.format(root=dataset_dir), 'r') as f:
        print('A2D color codes:')
        found = False
        label_colors = []
        for line in f.readlines():
            if found:
                name, iid, valid, R, G, B = line.rstrip().split()
                label_color = color_utils.ColorCode(name=name, id=int(iid), color=(int(R),int(G),int(B)))
                label_colors.append(label_color)
                print(label_color)
            found = found or all([kw in line for kw in ['NAME', 'ID', 'Valid','R','G','B']])
    color_utils.save_color_codes(label_colors, label_colors_json)
    for split in df_meta['split'].unique():
        df_split = df_meta[df_meta['split'] == split]
        df_split.drop(columns=['split'])
        df_split.to_csv(A2D_IMAGE_SPLIT_CSV.format(split=split), index=False)
    return


def kth_action_video_nobbox(kth_action_home):
    fpath = os.path.join(kth_action_home, '00sequences.txt')
    splits = {}
    df = dict(video=[], pid=[], action=[], split=[], fids=[])
    with open(fpath, 'r') as f:
        for line in f.readlines():
            if 'training:' in line.lower():
                splits.update({pid: 'train' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'validation:' in line.lower():
                splits.update({pid: 'val' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'test:' in line.lower():
                splits.update({pid: 'eval' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'frames' in line.lower():
                vid_file, kfs = line.strip().split('frames')
                vid_file = vid_file.strip()
                #fids = sum([frange.split('-') for frange in kfs.strip().split(', ')], [])
                #fids = [int(id) - 1 for id in fids]
                fid_ranges = [frange.split('-') for frange in kfs.strip().split(', ')]
                fids = []
                for start, end in fid_ranges:
                    fids += list(range(int(start), int(end)))
                df['fids'].append(';'.join(list(map(str, fids))))
                pid, action, var = vid_file.split('_')
                pid = pid.split('person')[1]
                df['pid'].append(int(pid))
                df['action'].append(action)
                vid_file_basepath = KTH_VIDEO_FILE.format(action=action, pid=pid, var=var)
                vid_file_path = os.path.join(kth_action_home, vid_file_basepath)
                df['video'].append(vid_file_path)
                df['split'].append(splits[pid])
    df = pd.DataFrame(df)
    for split in df['split'].unique():
        df_split = df[df['split'] == split]
        df_split.drop(columns=['split'])
        df_split.to_csv(KTH_SPLIT_CSV.format(split=split), index=False)
    return


UCF_VIDEO = 'video'
UCF_CLASS_IDX = 'classId'
UCF_CLIP_PATH = 'clip_path'
UCF_RECG_SPLIT_PATH = \
    '{dataset_dir}/ucfTrainTestlist/{split}list{no:02d}.txt'
UCF_RECG_CLS_INDEX_PATH = \
    '{dataset_dir}/ucfTrainTestlist/classInd.txt'
UCF_RECG_SPLIT_FILE_NO = {'train': 3, 'test': 3}
UCF_RECG_SPLIT_HEADERS = ['vid_path', UCF_CLASS_IDX]
UCF_RECG_TRAIN_SPLIT_CSV = 'data_model/ucf_recg_{split}.csv'
UCF_NUM_CLASSES = 101
INVALID_UCF_RECG_VIDS = ['PushUps/v_PushUps_g16_c04.avi',
                         'BlowingCandles/v_BlowingCandles_g05_c03.avi',
                         'SkyDiving/v_SkyDiving_g02_c01.avi',
                         'HorseRiding/v_HorseRiding_g14_c02.avi']


def ucf_recognition(dataset_dir, train_val_split_ratio=[0.95, 0.05]):
    class_index_fpath = UCF_RECG_CLS_INDEX_PATH.format(dataset_dir=dataset_dir)
    cls_ind = pd.read_csv(
        class_index_fpath, header=None, delimiter=' ', names=[UCF_CLASS_IDX, 'name'])
    cls_ind.set_index(UCF_CLASS_IDX, inplace=True)

    for split, num_split_files in UCF_RECG_SPLIT_FILE_NO.items():
        dfs = {'train': [], 'val': [], 'test':[]}
        for i in range(num_split_files):
            split_fpath = UCF_RECG_SPLIT_PATH.format(
                dataset_dir=dataset_dir, split=split, no=i+1)
            print(split_fpath)
            df_split = pd.read_csv(
                split_fpath, header=None, delimiter=' ', names=UCF_RECG_SPLIT_HEADERS)
            df_split = df_split[~df_split['vid_path'].isin(INVALID_UCF_RECG_VIDS)]
            df_split[UCF_CLIP_PATH] = df_split['vid_path'].apply(lambda x: os.path.join(dataset_dir, x))
            if split == 'train':
                df_split['className'] = df_split[UCF_CLASS_IDX].apply(lambda x: cls_ind.loc[x]['name'])
                df_split['split'] = df_split.apply(lambda x: np.random.choice(
                    ['train', 'val'], p=train_val_split_ratio), axis=1)
                dfs['train'].append(df_split[df_split['split'] == 'train'])
                dfs['val'].append(df_split[df_split['split'] == 'val'])
            else:
                dfs[split].append(df_split)
        for split, df_list in dfs.items():
            if df_list:
                df = pd.concat(df_list).reset_index()
                duplicated_rows = df[df.duplicated(subset=['vid_path'], keep='first')]
                df = df.drop(duplicated_rows.index)
                df.to_csv(UCF_RECG_TRAIN_SPLIT_CSV.format(split=split), index=False)
    return

