#!/bin/bash

# Directory path containing .mp4 files
A2D_CLIPS_PATH="/home/shuangludai/A2D/A2D_main_1_0/Release/clips320H/*.mp4"
A2D_CLIPS_OUTPUT_DIR="/home/shuangludai/A2D/A2D_main_1_0/Release/SAMclips320H"
SAM_MODEL_DIR="/home/shuangludai/ai_models/segment_anything"
mkdir -p $A2D_CLIPS_OUTPUT_DIR
python3 tools/sam_video_segments.py\
  --sam_model_dir=$SAM_MODEL_DIR\
  --sam_model_type="vit_b"\
  --clips_path=$A2D_CLIPS_PATH\
  --output_clips_dir=$A2D_CLIPS_OUTPUT_DIR