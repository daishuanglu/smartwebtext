import glob
from tqdm import tqdm
import os
import pandas as pd


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