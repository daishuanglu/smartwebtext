import argparse
import requests
import os
import wget
import shutil
import sys
import zipfile

KTH_VIDEOS_URLS = [
    'http://www.csc.kth.se/cvap/actions/walking.zip',
    'http://www.csc.kth.se/cvap/actions/jogging.zip',
    'http://www.csc.kth.se/cvap/actions/running.zip',
    'http://www.csc.kth.se/cvap/actions/boxing.zip',
    'http://www.csc.kth.se/cvap/actions/handwaving.zip',
    'http://www.csc.kth.se/cvap/actions/handclapping.zip']

KTH_METADATA_URL = 'https://www.csc.kth.se/cvap/actions/00sequences.txt'
KTH_DATASET_DIR = 'data_model/kth_action_dataset'


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

#create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for KTH action dataset')
    args = parser.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)
    for url in KTH_VIDEOS_URLS:
        print(url)
        filename = url.split('/')[-1]
        if not os.path.exists(filename):
            filename = wget.download(url, bar=bar_progress)
        sav_dir = os.path.join(args.dataset_dir, filename.split('.zip')[0])
        os.makedirs(sav_dir, exist_ok=True)
        print('Unzipping to ', sav_dir)
        with zipfile.ZipFile(filename , 'r') as zip_ref:
            zip_ref.extractall(sav_dir)
        print('Remove cached zipfile', filename)
        os.remove(filename)
    filename = wget.download(KTH_METADATA_URL)
    sav_path = os.path.join(args.dataset_dir, KTH_METADATA_URL.split('/')[-1])
    print('moving to', sav_path)
    shutil.move(filename, sav_path)

