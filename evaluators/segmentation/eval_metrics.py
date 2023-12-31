# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

from collections import defaultdict
import itertools
import json
import pandas as pd
from typing import List, Dict
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from utils import metric_utils
from preprocessors.video_annotation import VideoMetadata
from preprocessors.pipelines import VID_SEG_TRAIN_SPLIT_CSV


def db_eval_sequence(pred_label_masks: List[np.array], 
                     gt_label_masks: List[np.array], 
                     measure='J'):
  """
  Evaluate video sequence results.

	Arguments:
		segmentations (list of ndarrya): segmentations masks.
		annotations   (list of ndarrya): ground-truth  masks.
    measure       (char): evaluation metric (J,F,T)
    n_jobs        (int) : number of CPU cores.

  Returns:
    results (list): ['raw'] per-frame, per-object sequence results.
  """

  results = {'raw': []}
  for pred_label_mask, gt_label_mask in zip(pred_label_masks, gt_label_masks):
    obj_ids = np.unique(gt_label_mask)
    for obj_id in obj_ids:
      if obj_id != 0:
        obj_metric = metric_utils.segmentation_db_measures[measure](
          pred_label_mask==obj_id, gt_label_mask==obj_id)
        results['raw'].append(obj_metric)
    
  for stat, stat_fn in metric_utils.segmentation_statistics.items():
    results[stat] = [float(stat_fn(r)) for r in results['raw']]
  return results


def db_eval(pred_fpath, list_of_gt_meta: List[Dict], unique_cls_id_map, measures):
  """
  Evaluate video sequence results.

	Arguments:
		segmentations (list of ndarrya): segmentations masks.
		annotations   (list of ndarrya): ground-truth  masks.
    measure       (char): evaluation metric (J,F,T)
    n_jobs        (int) : number of CPU cores.

  Returns:
    results (dict): [sequence]: per-frame sequence results.
                    [dataset] : aggreated dataset results.
  """
  s_eval = defaultdict(dict)  # sequence evaluation
  d_eval = defaultdict(dict)  # dataset  evaluation
  for gt_meta in tqdm(list_of_gt_meta, desc='eval GT metadata'):
    fids = VideoMetadata.frame_ids(gt_meta)
    vid = VideoMetadata.video_id(gt_meta)
    gt_contexts = VideoMetadata.frame_annotations(gt_meta, unique_cls_id_map)
    prediction = VideoMetadata.load_sequence_predictions(pred_fpath.format(vid=vid))
    for fid, gt in zip(fids, gt_contexts):
      i = fids.index(fid)
      pred_label_mask, pred_obj_masks = VideoMetadata.prediction_to_masks(
        prediction['prob'][i], 
        prediction['cls_prob'][i], 
        prediction['min_conf'], 
        prediction['bg_conf'])
      for measure in measures:
        s_eval[vid][measure] = db_eval_sequence([pred_label_mask], [gt.label_mask], measure)
    
    for statistic in metric_utils.segmentation_statistics.keys():
      raw_data = np.hstack([s_eval[sequence][measure][statistic] for sequence in
        s_eval.keys()])
      d_eval[measure][statistic] = float(np.mean(raw_data))

  g_eval = {'sequence':dict(s_eval), 'dataset':dict(d_eval)}

  return g_eval


def print_results(evaluation, method_name="-"):
  """Print result in a table."""
  metrics = evaluation['dataset'].keys()
  # Print results
  table = PrettyTable(['Method']+[p[0]+'_'+p[1] for p in
    itertools.product(metrics, metric_utils.segmentation_statistics.keys())])
  table.add_row([method_name]+["%.3f"%np.round(
    evaluation['dataset'][metric][statistic],3) for metric, statistic in
    itertools.product(metrics, metric_utils.segmentation_statistics.keys())])
  print("\n{}\n".format(str(table)))


if __name__ == '__main__':
  PREDICTIONS_HDFS_PATH = 'evalution/vid_seg_predictions/pix2pix_video_segmentor/{vid}.h5'
  GT_METADATA_CSV = VID_SEG_TRAIN_SPLIT_CSV.format(split='test')
  OUTPUT_PATH = 'evalution/vid_seg_predictions/mertrics_pix2pix_video_segmentor.json'
  df = pd.read_csv(
    GT_METADATA_CSV, dtype=str, parse_dates=False, na_values=[], keep_default_na=False)
  gt_meta = [row.to_dict() for _, row in df.iterrows()]
  unique_cls_id_map = VideoMetadata.unique_class_id_map(split='')
  g_eval = db_eval(PREDICTIONS_HDFS_PATH, gt_meta, unique_cls_id_map, measures=[''])
  print_results(g_eval)
  with open(OUTPUT_PATH, 'w') as fp:
    json.dump(obj=g_eval, fp=fp)
  