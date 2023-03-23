import os
import pims
import pandas as pd
from transformers import pipeline
from PIL import Image
from tqdm import tqdm


OBJ_DETECTOR = pipeline(model="facebook/detr-resnet-50")
ZS_OBJ_DETECTOR = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")

# download and unzip KTH action videos from here https://www.csc.kth.se/cvap/actions/
KTH_ACTION_HOME = 'C:\\Users\\shud0\\KTHactions'
KTH_VIDEO_FILE = '{action}/person{pid}_{action}_{var}_uncomp.avi'
KTH_ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

def get_synthetic_bboxes(vid_file_path, fids, labels=['person']):
    v = pims.Video(vid_file_path)
    df_data = pd.DataFrame([], columns=['fid', 'x', 'y', 'w', 'h', 'confidence'], index=fids)
    for fid in tqdm(fids, desc='frame', position=1,  leave=False):
        _fid = int(fid)
        #print(fid, len(v))
        if _fid >= len(v):
            break
        img = Image.fromarray(v[_fid])
        #results = ZS_OBJ_DETECTOR(img, candidate_labels=labels)
        results = OBJ_DETECTOR(img)
        results = [res for res in results if res['label'] in labels]
        if not results:
            continue
        for res in results:
            df_data.at[fid, 'fid'] = _fid
            x,y,w,h = res['box']['xmin'], res['box']['ymin'],\
                res['box']['xmax']-res['box']['xmin'], res['box']['ymax']-res['box']['ymin']
            df_data.at[fid, 'x'] = x / img.width
            df_data.at[fid, 'y'] = y / img.height
            df_data.at[fid, 'w'] = w / img.width
            df_data.at[fid, 'h'] = h / img.height
            df_data.at[fid, 'height'] = img.height
            df_data.at[fid, 'width'] = img.width
            df_data.at[fid, 'confidence'] = res['score']
    return df_data


def parse_kth_splits():
    fpath = os.path.join(KTH_ACTION_HOME, '00sequences.txt')
    splits = {}
    dfs = []
    with open(fpath, 'r') as f:
        for line in tqdm(f.readlines(), desc='KTH bbox synthetic data', position=0,  leave=False):
            if 'training:' in line.lower():
                splits.update({pid: 'train' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'validation:' in line.lower():
                splits.update({pid: 'val' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'test:' in line.lower():
                splits.update({pid: 'val' for pid in line.strip().split('person')[1].strip().split(', ')})
            if 'frames' in line.lower():
                vid_file, kfs = line.strip().split('frames')
                vid_file = vid_file.strip()
                #fids = sum([frange.split('-') for frange in kfs.strip().split(', ')], [])
                #fids = [int(id) - 1 for id in fids]
                fid_ranges = [frange.split('-') for frange in kfs.strip().split(', ')]
                fids = []
                for start, end in fid_ranges:
                    fids += list(range(int(start), int(end)))
                pid, action, var = vid_file.split('_')
                pid = pid.split('person')[1]
                for var in ['d1', 'd2', 'd3', 'd4']:
                    vid_file_path = os.path.join(
                        KTH_ACTION_HOME, KTH_VIDEO_FILE.format(action=action, pid=pid, var=var))
                    if os.path.exists(vid_file_path):
                        #print(vid_file_path)
                        df_bbox = get_synthetic_bboxes(vid_file_path, fids, labels=['person'])
                        df_bbox = df_bbox.reset_index()
                        df_bbox['pid'] = pid
                        df_bbox['action'] = action
                        df_bbox['split'] = splits[pid]
                        df_bbox['video_path'] = vid_file_path
                        dfs.append(df_bbox)
    df_data = pd.concat(dfs)
    return df_data


if __name__== '__main__':
    kth_data = parse_kth_splits()
    os.makedirs('data_model')
    kth_data.to_csv('data_model/kth_actions.csv', index=False)
