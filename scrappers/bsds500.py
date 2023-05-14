import argparse
from utils import download_utils

BSDS_URL = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Output directory for BSDS image segementation dataset')
    args = parser.parse_args()
    download_utils.zip(BSDS_URL, args.dataset_dir, 'tar')