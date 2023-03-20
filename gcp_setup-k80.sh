#!/bin/bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install unzip
sudo apt-get install tmux
sudo apt-get install python3-pip
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
#sudo rm -r /usr/bin/nvidia-smi
sudo python3 install_gpu_driver.py
nvidia-smi
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114
mkdir $1
mkdir getwebtext/fasttext
mkdir getwebtext/data_model