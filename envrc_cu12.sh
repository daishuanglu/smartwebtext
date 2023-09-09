#!/bin/bash
set -eux
#pip install virtualenv
#virtualenv venv
#source venv/bin/activate
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip3 install nltk
python3 -m nltk.downloader stopwords
python3 -m nltk.downloader wordnet
python3 -m nltk.downloader omw-1.4
pip3 install spacy
python3 -m spacy download en_core_web_sm
pip install git+https://github.com/facebookresearch/segment-anything.git
pip3 install -U -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.