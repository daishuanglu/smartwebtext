#!/bin/bash
# choose GCP the second image for Debian T4 cuda 11.4
#Welcome to the Google Deep Learning VM
#Version: common-cu113.m98
#Based on: Debian GNU/Linux 10 (buster) (GNU/Linux 4.19.0-21-cloud-amd64 x86_64\n)
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install unzip
sudo apt-get install tmux
#sudo apt-get install python3-pip
nvidia-smi
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
mkdir $1
mkdir /home/shuangludai/$1/fasttext
mkdir /home/shuangludai/$1/data_model
mkdir /home/shuangludai/$1/data_model/model
strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX