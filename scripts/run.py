#!/usr/bin/env python
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.connectors.profiler_connector import PROFILERS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ifitb.data import data_module
from ifitb.data.data_module import IntentionFitbDataModule
from ifitb.model.t5_filler_model import T5FillerModel
from ifitb.model.t5_visual_module import T5AndVisual
from ifitb.util.argparse_with_defaults import ArgumentParserWithDefaults


def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()

    parser.add_argument("--reasons-by-verb-path", default=data_module.PATH_REASONS_BY_VERB)
    parser.add_argument("--data-path", default=data_module.PATH_DATA)
    # parser.add_argument("--visual-data-dir", default="data/I3D_video_features")  # TODO

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", "-j", type=int, default=0,
                        help="data loader workers. Each worker batch-tokenizes in parallel, "
                             "so maybe don't make this number equal to the number of CPU cores but just a small "
                             "natural number.")

    parser.add_argument("--gpus", type=int)
    parser.add_argument("--visual-size", type=int, default=1024)

    # The only models that work with the used pipelines are the ones from `MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING`.
    # The model config names can't be obtained easily. You can obtain all the officially supported ones, of all types,
    # but then it's hard to know which ones are in this list.
    # Also, note you still can't easily get the user-uploaded models, as they're resolved dynamically.
    # So we can't provide model name choices.
    # I guess we can check the options from the URL below, though I'm not sure if that's the exact filter tag.
    parser.add_argument("--model", default="t5-base",
                        help="pipeline model. Check the options in https://huggingface.co/models?filter=seq2seq")

    parser.add_argument("--text-only", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")

    parser.add_argument("--predictions-output-path", default="predictions.csv")

    parser.add_argument("--trainer-default-root-dir")

    parser.add_argument("--train", action="store_true")
    # parser.add_argument("--test", action="store_true")  # TODO: to use test set
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--profiler", choices=PROFILERS)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-scheduler", choices=["", "linear_with_warmup"], default="linear_with_warmup",
                        type=lambda s: s or None)
    parser.add_argument("--weight-decay", default=1e-4, type=float)

    return parser.parse_args()


def _pandas_float_format(x: float) -> str:
    if x == 0:
        return "0"
    elif abs(x) < 0.005:
        return np.format_float_scientific(x, exp_digits=1, precision=0, trim="-")
    else:
        return f"{x:.2f}"


def main() -> None:
    args = _parse_args()

    pl.seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.text_only:
        t5_like_pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    else:
        t5_like_pretrained_model = T5AndVisual.from_pretrained(args.model, visual_size=args.visual_size)

    filler = T5FillerModel(t5_like_pretrained_model=t5_like_pretrained_model, tokenizer=tokenizer, lr=args.lr,
                           lr_scheduler=args.lr_scheduler, weight_decay=args.weight_decay)

    trainer = pl.Trainer(gpus=args.gpus, default_root_dir=args.trainer_default_root_dir, fast_dev_run=args.fast_dev_run,
                         max_epochs=args.epochs, benchmark=args.benchmark, deterministic=args.deterministic,
                         profiler=args.profiler)

    data_module = IntentionFitbDataModule(tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers,
                                          output_visual=not args.text_only,
                                          reasons_by_verb_path=args.reasons_by_verb_path, data_path=args.data_path)

    if args.train:
        trainer.fit(filler, datamodule=data_module)

    trainer.test(filler, datamodule=data_module)

    predictions = {k: v.tolist() if isinstance(v, torch.Tensor) else v
                   for k, v in next(iter(trainer.evaluation_loop.predictions.predictions.values())).items()}
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(args.predictions_output_path, index=False)
    print(f"Predictions saved in {args.predictions_output_path}. First rows:")
    print()
    pd.options.display.float_format = _pandas_float_format
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 0,
                           "display.max_colwidth", None):
        print(df.head(10))


if __name__ == "__main__":
    main()
