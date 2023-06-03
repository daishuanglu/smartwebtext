import argparse
import glob
import os.path

from tools import sam_client

CLIPS_DIR="/home/shuangludai/A2D/A2D_main_1_0/Release/clips320H"
OUTPUT_CLIPS_DIR="/home/shuangludai/A2D/A2D_main_1_0/Release/SAMclips320H"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sam_model_dir', required=True,
        help='Segment anything local model directory.')
    parser.add_argument(
        '--sam_model_type', default='vit-b',
        help='Segment anything model type (vit-b/vit-l/vit-h).')
    parser.add_argument(
        '--clips_path', required=True,
        help='Input video clips path patterns')
    parser.add_argument(
        '--output_clips_dir', required=True, help='Output video clips directory')
    args = parser.parse_args()
    client = sam_client.SAMClient(model_dir=args.sam_model_dir, model_type=args.sam_model_type)
    for video_path in glob.glob(args.clips_path):
        output_fname = os.path.basename(video_path)
        segmented_video_path = os.path.join(args.output_clips_dir, output_fname)
        client.segment_video(video_path, segmented_video_path)

