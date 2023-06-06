import math
import numpy as np
import os
from typing import Dict, List
import pandas as pd
import glob
from tqdm import tqdm
from utils import string_utils
from utils import color_utils
from preprocessors import dataloader
import time

PRNEWS_FILEPATTERN = 'data_model/scrapped_news/webtext_thread_*.txt'
PRNEWS_SEPARATOR = '\t'
PRNEWS_HEADERS = ['Company', 'Stock', 'date', 'Title', 'Body']
PRNEWS_INVALID_KEYTERMS = ['follow us', 'twitter|', 'linkedin|', '.copyright',
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
    stopwords = string_utils.load_stopwords()
    s = s.lower()
    s = [w.strip() for w in s.split(PRNEWS_PARAGRAPH_SEP)]
    for term in PRNEWS_INVALID_KEYTERMS:
        s = list(filter(lambda x: term not in x, s))
    s = [string_utils.remove_punct(sp) for sp in s]
    s = list(filter(lambda x: len(x.split()) > PRNEWS_SENTENCE_MIN_WORDS, s))
    s = [string_utils.clean_char(sp) for sp in s]
    s = [string_utils.remove_stopwords(sp, stopwords, sub=' ') for sp in s]
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
            return PRNEWS_PARAGRAPH_SEP.join(
                [row['Title'].lower(), row['Body'].lower()]).strip()

    textfile = dataloader.BaseTextFile(
        fpath=PRNEWS_FILEPATTERN, sep=PRNEWS_SEPARATOR,
        types={'Body': _body, 'Company': _company, 'Title': _title},
        col_fns=[('Text', _text)], must_contain=PRNEWS_MUST_CONTAIN_COLS)
    textfile.write(
        output_files=output_files,
        split_ratio=split_ratio,
        cols=['Company', 'Title', 'Text'],
        explode_sep=PRNEWS_PARAGRAPH_SEP,
        sep=PRNEWS_DATA_SEP)
    for vocab_path, vocab_cols in vocabs.items():
        textfile.vocab(vocab_cols, vocab_path)


def prnews_negatives(data_path,  ref_col, score_col, neg_ratio=1.0):
    print('add negative samples for %s' % data_path)
    df = pd.read_csv(
        data_path, sep=PRNEWS_DATA_SEP,
        dtype=str, keep_default_na=False, na_values=[], parse_dates=False)
    df[score_col] = 1.0
    df_uniq_refs = df[ref_col].value_counts()
    print('unique reference values')
    print(df_uniq_refs)
    os.remove(data_path)
    start_time = time.time()
    for i, (ref, counts) in enumerate(df_uniq_refs.iteritems()):
        n_neg_samples = int(math.ceil(counts * neg_ratio))
        df_pos = df[df[ref_col] == ref]
        df_negs = df[df[ref_col] != ref]
        df_neg_sampled = df_negs.sample(n=n_neg_samples, replace=n_neg_samples<len(df_negs))
        df_neg_sampled[score_col] = 0.0
        pd.concat([df_pos, df_neg_sampled]).to_csv(
            data_path, sep=PRNEWS_DATA_SEP, mode='w' if i==0 else 'a', index=False, header=(i == 0))
        print('%d/%d %s, %d secs.' % (i+1, len(df_uniq_refs), ref, int(time.time()-start_time)))
    return


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


def kth_action_video():
    textfile = dataloader.PandasTextFile(fpath=KTH_ACTION_SRC_CSV, sep=KTH_ACTION_DATA_CSV_SEP)
    df = textfile.all()
    for split in df[KTH_SPLIT_COL].unique():
        df_split = df[df[KTH_SPLIT_COL] == split]
        df_output = pd.DataFrame([], index=df_split['vid'].unique())
        for uuid in df_split['vid'].unique():
            df_split_uuid = df_split[df_split['vid'] == uuid]
            df_split_uuid = df_split_uuid.sort_values(
                ['fid', 'x', 'y'], ascending=[True, True, True])
            for col in df_split.columns:
                df_output.at[uuid, col] = df_split_uuid[col].str.cat(sep=KTH_FRAME_FEATURE_SEP)
        df_output.to_csv(KTH_SPLIT_CSV.format(split=split), index=False)
        print('KTH action ', split, 'set: ', len(df), 'samples')
    return


def eval_example_sentences(
        fpath, text_cols, row_delimiter, multi_sent_seps, preproc_sep, preproc_fn=None):

    def _text_col(s):
        s = [w.strip() for w in string_utils.split_by_multi_sep(s, multi_sent_seps)]
        s = [w for w in s if w]
        if preproc_fn is None:
            return preproc_sep.join(s)
        else:
            return preproc_fn(preproc_sep.join(s))

    textfile = dataloader.PandasTextFile(
        fpath=fpath, sep=row_delimiter, types={col: _text_col for col in text_cols})
    texts = textfile.all()[text_cols].dropna().astype(str).apply(preproc_sep.join, axis=1)
    texts = texts.str.split(preproc_sep).explode()
    texts.replace('', math.nan, inplace=True)
    texts = texts.dropna().tolist()
    return texts