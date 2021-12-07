# Main method of "WhyAct: Identifying Action Reasons in Lifestyle Vlogs"

This is the main method of the paper "WhyAct: Identifying Action Reasons in Lifestyle Vlogs" by Ignat et al., 2021.
See more info in the [main repo](https://github.com/michigannlp/vlog_action_reason).

Some code in this repo is based on the [Video Fill-in-the-Blank
code](https://github.com/MichiganNLP/video-fill-in-the-blank).

## 1. Setup

Using Conda:

```bash
conda env create
conda activate intention-fitb
```

Place the JSON files under `data/`.

## 2. Download the videos

This step is only necessary if you want to extract the features, which you don't have to do as we provide them already.

a. Install [youtube-dl](https://youtube-dl.org/), [FFmpeg](https://www.ffmpeg.org/),
   [jq](https://stedolan.github.io/jq/), and [tqdm](https://github.com/tqdm/tqdm) (they can all be installed from 
   Conda).
b. Run [`download_videos.sh` from the main
   repo](https://github.com/MichiganNLP/vlog_action_reason/blob/master/download_videos.sh).

## 3. Extract features

We provide pre-extracted features, and they're directly used from the code (you don't have to download them).
Still, if you want to run this step yourself, do:

```bash
./scripts/extract_features.py $INPUT_VIDEOS_FOLDER $OUTPUT_FEATURES_FOLDER
```

Use `--help` to see all the options.

## 4. Train

Follow these steps to train a new model. Note you don't have to do this as we provide [a pre-trained
model](https://www.dropbox.com/s/m0x6ey65jzjzgwz/intention-pretrained.ckpt?dl=1).

0. Split the val data into val and train (goes before or after the next one?).
a. Prepare the unlabeled data:

   ```bash
   ./scripts/fitb_data_without_val_test.py 
   ```

b. Fine-tune T5 on text+video on unlabeled data (you can do text-only).

   ```bash
   ./scripts/run.py --fitb-train
   ```

   You can see the available options using `--help`.

c. Fine-tune the obtained model on the val data.

   ```bash
   ./scripts/run.py --train
   ```

## 5. Evaluate

```bash
./scripts/run.py --checkpoint-path $CHECKPOINT_PATH --use-test-set
```

Feel free to try it with our pre-trained model:

```bash
CHECKPOINT_PATH=https://www.dropbox.com/s/m0x6ey65jzjzgwz/intention-pretrained.ckpt?dl=1
```
