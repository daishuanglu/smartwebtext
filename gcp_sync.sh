#!/bin/bash
set -eux
fdir=${2:-"smartwebtext"}
IP="35.185.212.101"
a="code"
b="prnews"
c="fasttext"
d='predictions'
e="setup"
f="model"
g="log"
if [ $1 == $g ]
then
  echo "download lightning logs ..."
  LOGDIR_LOCAL=${3:-"lightning_logs"}_sync
  LOGDIR=${3:-"lightning_logs"}
  mkdir -p $LOGDIR_LOCAL
  scp -r shuangludai@$IP:/home/shuangludai/$fdir/$LOGDIR/* $LOGDIR_LOCAL
fi
if [ $1 == $f ]
then
  echo "upload pretrained model ..."
  scp -r $3 shuangludai@$IP:/home/shuangludai/$fdir
fi
if [ $1 == $e ]
then
  echo "upload gcp setup scripts ..."
  scp -r gcp_setup*.sh shuangludai@$IP:/home/shuangludai
fi
if [ $1 == $a ]
then
  echo "update code to google cloud ..."
  scp -r scrappers shuangludai@$IP:/home/shuangludai/$fdir
  scp -r utils shuangludai@$IP:/home/shuangludai/$fdir
  scp -r models shuangludai@$IP:/home/shuangludai/$fdir
  scp -r preprocessors shuangludai@$IP:/home/shuangludai/$fdir
  scp -r experiments shuangludai@$IP:/home/shuangludai/$fdir
  scp requirements.txt shuangludai@$IP:/home/shuangludai/$fdir
  scp envrc.sh shuangludai@$IP:/home/shuangludai/$fdir
  scp -r config shuangludai@$IP:/home/shuangludai/$fdir
  scp -r svo shuangludai@$IP:/home/shuangludai/$fdir
  #scp -r evaluation shuangludai@$IP:/home/shuangludai/$fdir
  scp -r professional shuangludai@$IP:/home/shuangludai/$fdir
  scp -r tools shuangludai@$IP:/home/shuangludai/$fdir
  scp env_setup.sh shuangludai@$IP:/home/shuangludai/$fdir
fi
if [ $1 == $b ]
then
  echo "update prnews dataset ..."
  cd data_model
  tar -czvf _tmp_data.gz scrapped_news
  scp -r _tmp_data.gz shuangludai@$IP:/home/shuangludai/$fdir/data_model
  rm _tmp_data.gz
  cd ..
fi
if [ $1 == $c ]
then
    echo "update fasttext model ..."
    scp -r ig_fasttext/*en*100*.zip shuangludai@$IP:/home/shuangludai/$fdir/fasttext
    scp -r ig_fasttext/lid.176.bin shuangludai@$IP:/home/shuangludai/$fdir/fasttext
fi
if [ $1 == $d ]
then
    echo "download predictions ..."
    mkdir -p evaluation
    scp -r shuangludai@$IP:/home/shuangludai/$fdir/evaluation/* evaluation
fi