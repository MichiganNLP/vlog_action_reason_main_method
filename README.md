# Main method of "WhyAct: Identifying Action Reasons in Lifestyle Vlogs"

This is the main method of the paper "WhyAct: Identifying Action Reasons in Lifestyle Vlogs" by Ignat et al., 2021.
See more info in the [main repo](https://github.com/michigannlp/vlog_action_reason).

## Setup

Using Conda:

```bash
conda env create
conda activate intention-fitb
```

Place the JSON files under `data/`.

## Download the videos

This step is only necessary if you want to extract the features, which you don't have to do as we provide them already.

1. Install [youtube-dl](https://youtube-dl.org/), [FFmpeg](https://www.ffmpeg.org/),
   [jq](https://stedolan.github.io/jq/), and [tqdm](https://github.com/tqdm/tqdm) (they can all be installed from 
   Conda).
2. Run [`download_videos.sh` from the main
   repo](https://github.com/MichiganNLP/vlog_action_reason/blob/master/download_videos.sh).

## Extract features

We provide pre-extracted features, and they're directly used from the code (you don't have to download them).
Still, if you want to run this step yourself, do:

```bash
./scripts/extract_features.py $INPUT_VIDEOS_FOLDER $OUTPUT_FEATURES_FOLDER
```

Use `--help` to see all the options.

## Train

```bash
./scripts/run.py --train
```

See the available options using `--help`.
