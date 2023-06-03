import argparse
import glob
import os

from tools import sam_client


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sam_model_dir', required=True,
        help='Segment anything local model directory.')
    parser.add_argument(
        '--sam_model_type', default='vit_b',
        help='Segment anything model type ("vit_b"/"vit_l"/"vit_h").')
    parser.add_argument(
        '--clips_path', required=True,
        help='Input video clips path patterns')
    parser.add_argument(
        '--output_clips_dir', required=True, help='Output video clips directory')
    args = parser.parse_args()
    client = sam_client.SAMClient(model_dir=args.sam_model_dir, model_type=args.sam_model_type)
    clips_paths = glob.glob(args.clips_path)
    for i, video_path in enumerate(clips_paths):
        print('%d/%d, processing %s ' % (i, len(clips_paths), video_path))
        output_fname = os.path.basename(video_path)
        segmented_video_path = os.path.join(args.output_clips_dir, output_fname)
        client.segment_video(video_path, segmented_video_path)

