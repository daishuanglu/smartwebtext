#!/bin/bash
#python -m venv env
#source env/bin/activate
pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install nltk
python3 -m nltk.downloader stopwords
python3 -m nltk.downloader wordnet
python3 -m nltk.downloader omw-1.4
pip3 install spacy
python3 -m spacy download en_core_web_sm
pip3 install -U -r requirements.txt
