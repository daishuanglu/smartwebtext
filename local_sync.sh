#!/bin/bash
set -eux
fdir=${2:-"smartwebtext"}
IP="192.168.0.26"
a="code"
b="prnews"
c="fasttext"
d='val_pred'
e="setup"
f="model"
g="log"
if [ $1 == $g ]
then
  echo "download lightning logs to shud@192.168.0.XX..."
  LOGDIR_LOCAL=${3:-"lightning_logs"}_sync
  LOGDIR=${3:-"lightning_logs"}
  mkdir -p $LOGDIR_LOCAL
  scp -r shud@$IP:/home/shud/$fdir/$LOGDIR/* $LOGDIR_LOCAL
fi
if [ $1 == $f ]
then
  echo "upload pretrained model to shud@192.168.0.XX..."
  scp -r $3 shud@$IP:/home/shud/$fdir
fi
if [ $1 == $e ]
then
  echo "upload dev env setup scripts to shud@192.168.0.XX..."
  scp -r setup.sh shud@$IP:/home/shud
fi
if [ $1 == $a ]
then
  echo "update code to shud@192.168.0.XX ..."
  scp -r scrappers shud@$IP:/home/shud/$fdir
  scp -r utils shud@$IP:/home/shud/$fdir
  scp -r models shud@$IP:/home/shud/$fdir
  scp -r preprocessors shud@$IP:/home/shud/$fdir
  scp -r experiments shud@$IP:/home/shud/$fdir
  scp -r evaluators shud@$IP:/home/shud/$fdir
  scp requirements.txt shud@$IP:/home/shud/$fdir
  scp -r env*.sh shud@$IP:/home/shud/$fdir
  scp -r config shud@$IP:/home/shud/$fdir
  scp -r svo shud@$IP:/home/shud/$fdir
  #scp -r evaluation shuangludai@$IP:/home/shuangludai/$fdir
  scp -r professional shud@$IP:/home/shud/$fdir
  scp -r sw_tools shud@$IP:/home/shud/$fdir
  scp -r modules shud@$IP:/home/shud/$fdir
  scp -r thin_plate_spline_motion_model shud@$IP:/home/shud/$fdir
  scp env_setup.sh shud@$IP:/home/shud/$fdir
fi
if [ $1 == $b ]
then
  echo "update prnews dataset to shud@192.168.0.XX..."
  cd data_model
  tar -czvf _tmp_data.gz scrapped_news
  scp -r _tmp_data.gz shud@$IP:/home/shud/$fdir/data_model
  rm _tmp_data.gz
  cd ..
fi
if [ $1 == $c ]
then
    echo "update fasttext model to shud@192.168.0.XX..."
    #scp -r fasttext/*en*100*.bin shuangludai@$IP:/home/shuangludai/$fdir/fasttext
    scp -r fasttext/lid.176.bin shud@$IP:/home/shud/$fdir/fasttext
fi
if [ $1 == $d ]
then
    echo "download validation predictions from shud@192.168.0.XX..."
    scp -r shud@$IP:/home/shud/$fdir/val_predictions ./
fi