import requests
import os
import re
import wget
import shutil
import sys
import zipfile
import tarfile
import rarfile
import gdown
from pytube import YouTube


GDRIVE_URL = 'https://drive.google.com/uc?id={ID}'


def download_clip_from_youtube(video_url, local_path):
    yt = YouTube(video_url)
    try:
        filtered_stream = yt.streams.filter(file_extension='mp4').first()
        filtered_stream.download(filename=local_path, timeout=3000, max_retries=3)
    except Exception as e:
        return {'status': e.__repr__()}
    
    return {'status': 'success'}

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def extract_google_drive_file_id(link):
    match = re.search(r"/d/([A-Za-z0-9_-]+)", link)
    # Extract the file ID from the match object
    file_id = match.group(1)
    return file_id

#create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
  progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
  # Don't use print() as it will print in new line every time.
  sys.stdout.write("\r" + progress_message)
  sys.stdout.flush()


def download_url(url, save_path, chunk_size=None):
    if chunk_size is not None:
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wget.download(url, bar=bar_progress, out=save_path)
        print(f"File downloaded to: {save_path}")

def dl(url, output_fname):
    filename = url.split('/')[-1] if output_fname is None else output_fname
    if url.startswith(r'https://drive.google.com/'):
        fid = extract_google_drive_file_id(url)
        print('google drive file ID=', fid)
        gdown.download(GDRIVE_URL.format(ID=fid), filename, quiet=False)
    else:
        filename = wget.download(url, bar=bar_progress)
    return filename


def metadata_txt(url, dataset_dir, output_fname=None):
    os.makedirs(dataset_dir, exist_ok=True)
    filename = dl(url, output_fname)
    if output_fname is None:
        sav_path = os.path.join(dataset_dir, url.split('/')[-1])
    else:
        sav_path = os.path.join(dataset_dir, output_fname)
    print('moving to', sav_path)
    shutil.move(filename, sav_path)


def unzip_file(filename, sav_dir, type='zip'):
    if type == 'zip':
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(sav_dir)
    elif type == 'gz':
        with tarfile.open(filename, 'r:gz') as tar:
            # Extract all files
            tar.extractall(sav_dir)
    elif type == 'bz':
        with tarfile.open(filename, 'r:bz2') as tar:
            # Extract all files
            tar.extractall(sav_dir)
    elif type == 'rar':
        with rarfile.RarFile(filename, "r") as rar_ref:
            rar_ref.extractall(sav_dir)
    else:
        print('Must specify a valid zip file type (gz/zip)!')


def zip(url, dataset_dir, type='zip', output_fname=None):
    os.makedirs(dataset_dir, exist_ok=True)
    filename = dl(url, output_fname)
    print('dowloaded file:', filename)
    if output_fname is None:
        sav_dir = os.path.join(dataset_dir, filename.split('.')[0])
    else:
        sav_dir = os.path.join(dataset_dir, output_fname.split('.')[0])
    #os.makedirs(sav_dir, exist_ok=True)
    print('Unzipping to ', dataset_dir)
    unzip_file(filename, dataset_dir, type)
    print('Remove cached zipfile', filename)
    os.remove(filename)
    return sav_dir


