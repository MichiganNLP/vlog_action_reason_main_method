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

## 2. Download the videos

This step is only necessary if you want to extract the features, which you don't have to do as we provide them already.

1. Install [youtube-dl](https://youtube-dl.org/), [FFmpeg](https://www.ffmpeg.org/),
   [jq](https://stedolan.github.io/jq/), and [tqdm](https://github.com/tqdm/tqdm) (they can all be installed from 
   Conda).
2. Run [`download_videos.sh` from the main
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
model](https://github.com/MichiganNLP/vlog_action_reason_main_method/releases/download/files/epoch.3-step.223_only_model.pt.zip).

1. Prepare the unlabeled data by removing the test data out of all the raw data:

  ```bash
  ./scripts/fitb_data_without_test.py \
    https://github.com/MichiganNLP/vlog_action_reason_main_method/releases/download/files/dict_sentences_per_verb_all_MARKERS.json \
    https://raw.githubusercontent.com/MichiganNLP/vlog_action_reason/master/data/test.json \
    > dict_sentences_per_verb_all_MARKERS_without_test.json
  ```

2. Fine-tune T5 on text+video on this unlabeled data (we already computed it, but you can set your own using 
`--fitb-data-path dict_sentences_per_verb_all_MARKERS_without_test.json`):

  ```bash
  ./scripts/run.py --fitb-train
  ```

  > You can see the available options using `--help`.

  The saved checkpoint will be in `CHECKPOINT_PATH=lightning_logs/version_$N/checkpoints/epoch=$E-step=$S.ckpt`.

3. Subsequently, fine-tune on the dev data and evaluate it on the test set.

  ```bash
  ./scripts/run.py --train --use-test-set --checkpoint-path $CHECKPOINT_PATH
  ```

  Similarly to the previous step, you can find a new checkpoint was created.

## 5. Evaluate

To just evaluate a checkpoint without training, do:

```bash
./scripts/run.py --use-test-set --checkpoint-path $CHECKPOINT_PATH
```

Feel free to try it out with our pre-trained model:

```bash
CHECKPOINT_PATH=https://github.com/MichiganNLP/vlog_action_reason_main_method/releases/download/files/epoch.3-step.223_only_model.pt.zip
```
