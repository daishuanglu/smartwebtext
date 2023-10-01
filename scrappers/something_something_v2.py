"""
Moving Objects Dataset: Something-Something v. 2
Your model recognizes certain simple, single-frame gestures like a thumbs-up. But for a truly responsive, accurate system, you want your model to recognize gestures in the context of everyday objects. Is the person pointing to something or wagging their index finger? Is the hand cleaning the display or zooming in and out of an image with two fingers? Given enough examples, your model can learn the difference.

The Something-Something dataset (version 2) is a collection of 220,847 labeled video clips of humans performing pre-defined, basic actions with everyday objects. It is designed to train machine learning models in fine-grained understanding of human hand gestures like putting something into something, turning something upside down and covering something with something.

Samples from the Something-Something dataset:
Dataset details
Total number of videos	220,847
Training Set	168,913
Validation Set	24,777
Test Set (w/o labels)	27,157
Labels	174
Quality	100px
FPS	12
The dataset was created with the help of more than 1,300 unique crowd actors.

In order to extract the TGZ archive, the 20 files must be downloaded and concatenated in the right
order. Only then they can be extracted. Unzip the downloaded zip files separately and then run the
given command in the documentation to extract all the video files.
1. unzip 20bn-something-something-v2-\??.zip
2. cat 20bn-something-something-v2-?? | tar -xvzf –
"""

import argparse
import shutil
import os
import glob

from utils import download_utils


PARTS_URLS = [
    'https://drive.google.com/file/d/1Z50_k-czgaO_zMsJPs9Ubx2UxGBP6_H8/view?usp=sharing',
    'https://drive.google.com/file/d/1KQAnr1jlEtQ_hH7bUwraoL6i-azCHc6v/view?usp=sharing',
    'https://drive.google.com/file/d/1HW6GzNRDhBEHijHAcff8we9uJsG-El2Z/view?usp=sharing',
    'https://drive.google.com/file/d/12gQkYWvstsCkI1wgH9GAC8vJiJ55zRQK/view?usp=sharing',
    'https://drive.google.com/file/d/1VuRJJDNZGB-wJ5w-ZvfberPfc6C0CGTh/view?usp=sharing',
    'https://drive.google.com/file/d/1ELp6loHAhtAnSTmK0Y6L1RDIq3EXAmrP/view?usp=sharing',
    'https://drive.google.com/file/d/1XF5252O816rwMahvCE2JN4DLP9Bgs38D/view?usp=sharing',
    'https://drive.google.com/file/d/1m3tywCivdqcrukIOkLIgWWis-JpuuHlo/view?usp=sharing',
    'https://drive.google.com/file/d/1cjw3nHv7Eci_DX5VPPN3BJbjCt9QeS5J/view?usp=sharing',
    'https://drive.google.com/file/d/1wddyFg_FnUDGi9yKevdwGZAASQsg1s8b/view?usp=sharing',
    'https://drive.google.com/file/d/1PhDdWe8k32fGrbjytrQiTN6AW2vH-2b6/view?usp=sharing',
    'https://drive.google.com/file/d/1jB8S_mbOgnkvtaK5krv-NE7Zusy-8_Rf/view?usp=sharing',
    'https://drive.google.com/file/d/1AkUOEaGZbHfMQ_ukDOv_Byx2fhFKqu8n/view?usp=sharing',
    'https://drive.google.com/file/d/1Vlsdr4WIYYnPIJ7QO80x7lZhlT3TArTY/view?usp=sharing',
    'https://drive.google.com/file/d/1WhSqoV-BoaimNeK21gqC9MBPCiBR8uCR/view?usp=sharing',
    'https://drive.google.com/file/d/1zMbIfGwBc5bFi6X-yeSuDC_1E2TNY8Eb/view?usp=sharing',
    'https://drive.google.com/file/d/1QO_em7GRVwfItZyE81oThh5Qj9f_yrFY/view?usp=sharing',
    'https://drive.google.com/file/d/1RWuZlni-b0AnaL0304XmH_E4EBi-jadc/view?usp=sharing',
    'https://drive.google.com/file/d/1QrY0WkdgeSHEZHdmcMth2H2gh2FelTxf/view?usp=sharing',
    'https://drive.google.com/file/d/1heW15GyZ-jDpBC1OGzp6t1ZugADB4isT/view?usp=sharing'
]
LABEL_URL = 'https://drive.google.com/file/d/1nMZtU59z31TsNqTKN_-wN9H1DIanzHJt/view?usp=sharing'
LABEL_FILENAME = 'something-something-label.zip'
INFO_URL = 'https://drive.google.com/file/d/1MNfj3tYUKxQ68wb0n6FwAETsRA1BtEr3/view?usp=drive_link'
INFO_FILENAME = 'something-something-info.pdf'
DS_PARTS_FILENAME = '20bn-something-something-v2-{part:02d}.zip'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for something-something-v2 dataset')
    args = parser.parse_args()
    #class args:
    #    dataset_dir = 'D:/video_datasets'
    os.makedirs(args.dataset_dir, exist_ok=True)
    download_utils.dl(LABEL_URL, output_fname=LABEL_FILENAME)
    download_utils.dl(INFO_URL, output_fname=INFO_FILENAME)
    shutil.move(INFO_FILENAME, os.path.join(args.dataset_dir, INFO_FILENAME))
    data_dir = os.path.join(args.dataset_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    download_utils.unzip_file(LABEL_FILENAME, args.dataset_dir, type='zip')
    for i, url in enumerate(PARTS_URLS):
        file_part = DS_PARTS_FILENAME.format(part=i)
        download_utils.zip(url, data_dir, output_fname=file_part, type='zip')
    download_utils.concat_files(glob.glob(data_dir),
                                os.path.join(args.dataset_dir, 'data.tar.gz'))
    download_utils.unzip_file(os.path.join(args.dataset_dir, 'data.tar.gz'),
                              args.dataset_dir,
                              type='gz')
    shutil.rmtree(data_dir)
    os.remove(os.path.join(args.dataset_dir, 'data.tar.gz'))

