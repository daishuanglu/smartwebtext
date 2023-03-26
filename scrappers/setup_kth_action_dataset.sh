#!/bin/bash
DATADIR='data_model/KTH_action_dataset'
python3 scrappers/download_kth_actions.py --dataset_dir=$DATADIR
python3 scrappers/kth_actions_metadata.py\
  --output_dir='data_model/kth_actions.csv'\
  --dataset_dir=$DATADIR