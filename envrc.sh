#!/bin/bash
set -eux
#if [ -e data_model/_tmp_data.gz ]
#then
#    cd data_model
#    tar -xzvf _tmp_data.gz
#    rm _tmp_data.gz
#    cd ..
#else
#    echo "No data zip file found"
#fi
pip install virtualenv
virtualenv venv
source venv/bin/activate
# For cuda 11.6
#pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# For CUDA 12.1
pip3 install torch torchvision torchaudio
pip3 install nltk
python3 -m nltk.downloader stopwords
python3 -m nltk.downloader wordnet
python3 -m nltk.downloader omw-1.4
pip3 install spacy
python3 -m spacy download en_core_web_sm
pip install git+https://github.com/facebookresearch/segment-anything.git
pip3 install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.