#!/usr/bin/env bash

#SBATCH --job-name=extract_features
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=20
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=mihalcea1
#SBATCH --partition=gpu

source scripts/great_lakes/init.source

python scripts/extract_features.py \
  --frames-per-clip 64 \
  --batch-size 8 \
  --num-workers 20 \
  --num-workers-video-clips 20 \
  /scratch/mihalcea_root/mihalcea1/shared_data/intention/miniclips/no_check/ \
  /scratch/mihalcea_root/mihalcea1/shared_data/intention/i3d_video_features/no_check/
