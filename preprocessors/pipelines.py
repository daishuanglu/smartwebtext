import math
import os
import pandas as pd
import glob
from tqdm import tqdm
from utils import string_utils
from preprocessors import dataloader


PRNEWS_FILEPATTERN = 'data_model/scrapped_news/webtext_thread_*.txt'
PRNEWS_SEPARATOR = '\t'
PRNEWS_HEADERS = ['Company', 'Stock', 'date', 'Title', 'Body']
PRNEWS_INVALID_KEYTERMS = ['follow us', 'twitter|', 'linkedin|', '.copyright',
    'facebook:', 'visit:', 'twitter:', 'for more information', 'click here', 'email', 'phone', 'logo']
PRNEWS_SENTENCE_MIN_WORDS = 5
PRNEWS_PARAGRAPH_SEP = ';;;;'
PRNEWS_DATA_SEP = '\t'
PRNEWS_MUST_CONTAIN_COLS = ['Text', 'Company']

KTH_ACTION_SRC_CSV = 'data_model/kth_actions.csv'
KTH_ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
KTH_ACTION_DATA_CSV_SEP = ','
KTH_SPLIT_COL = 'split'
KTH_SPLIT_CSV = 'data_model/kth_actions_{split}.csv'
KTH_FRAME_FEATURE_SEP = ';'
#KTH_ACTION_HOME = '/home/shuangludai/KTHactions'
KTH_VIDEO_FILE = '{action}/person{pid}_{action}_{var}_uncomp.avi'


BSDS_SAMPLES = '{root}/images/{split}/*.jpg'
BSDS_GT_FILE = '{root}/groundTruth/{split}/{iid}.mat'
BSDS_SPLIT_CSV = 'data_model/bsds_{split}.csv'


def prnews_text_preproc(s):
    stopwords = string_utils.load_stopwords()
    s = s.lower()
    s = [w.strip() for w in s.split(PRNEWS_PARAGRAPH_SEP)]
    for term in PRNEWS_INVALID_KEYTERMS:
        s = list(filter(lambda x: term not in x.lower(), s))
    s = [string_utils.remove_punct(sp) for sp in s]
    s = list(filter(lambda x: len(x.split()) > PRNEWS_SENTENCE_MIN_WORDS, s))
    s = [string_utils.clean_char(sp) for sp in s]
    s = [string_utils.remove_stopwords(sp, stopwords, sub=' ') for sp in s]
    return PRNEWS_PARAGRAPH_SEP.join(s).strip()


def prnews(output_files, split_ratio, vocab_path):

    def _body(s):
        return prnews_text_preproc(s)

    def _title(s):
        s = string_utils.get_title_name(s, sep='-').strip()
        return s

    def _company(s):
        return string_utils.getcompanyname(s, sep='-').strip()

    def _text(row):
        return PRNEWS_PARAGRAPH_SEP.join(
            [row['Title'].lower(), row['Body'].lower()]).strip()

    textfile = dataloader.BaseTextFile(
        fpath=PRNEWS_FILEPATTERN, sep=PRNEWS_SEPARATOR,
        types={'Body': _body, 'Company': _company, 'Title': _title},
        col_fns=[('Text', _text)], must_contain=PRNEWS_MUST_CONTAIN_COLS)
    textfile.write(
        output_files=output_files,
        split_ratio=split_ratio,
        cols=['Company', 'Text'],
        explode_sep=PRNEWS_PARAGRAPH_SEP,
        sep=PRNEWS_DATA_SEP)
    textfile.vocab(['Company'], vocab_path)
    return


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