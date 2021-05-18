#!/usr/bin/env python
import argparse
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.connectors.profiler_connector import PROFILERS
from transformers import AutoTokenizer

from ifitb.data.alignment_data_module import ALIGNMENT_TRAIN_PATH, ALIGNMENT_VAL_PATH, AlignmentDataModule, \
    VISUAL_DATA_PATH
from ifitb.model.alignment_model import AlignmentModel
from ifitb.model.t5_visual_module import T5AndVisual
from ifitb.util.argparse_with_defaults import ArgumentParserWithDefaults


def _parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()

    parser.add_argument("--alignment-train-path", default=ALIGNMENT_TRAIN_PATH)
    parser.add_argument("--alignment-val-path", default=ALIGNMENT_VAL_PATH)
    parser.add_argument("--visual-data-dir", default=VISUAL_DATA_PATH)

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
    parser.add_argument("--checkpoint-path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")

    parser.add_argument("--trainer-default-root-dir")

    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--profiler", choices=PROFILERS)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr-scheduler", choices=["", "linear_with_warmup"], type=lambda s: s or None)
    parser.add_argument("--weight-decay", default=1e-4, type=float)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pl.seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    warnings.filterwarnings("ignore", message=r"Some weights of T5AndVisual .+ are newly initialized:"
                                              r" \['encoder\.embed_video\.\w+', 'encoder\.embed_video\.\w+'\]\n"
                                              r".+")  # FIXME: not working
    model = T5AndVisual.from_pretrained(args.model, visual_size=args.visual_size)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("encoder.embed_video")

    kwargs = {"model": model, "lr": args.lr, "lr_scheduler": args.lr_scheduler, "weight_decay": args.weight_decay}

    if args.checkpoint_path:
        alignment_model = AlignmentModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path, **kwargs)
    else:
        alignment_model = AlignmentModel(**kwargs)

    checkpoint_callback = ModelCheckpoint(every_n_train_steps=500, save_top_k=3)
    trainer = pl.Trainer(gpus=args.gpus, default_root_dir=args.trainer_default_root_dir, fast_dev_run=args.fast_dev_run,
                         benchmark=args.benchmark, deterministic=args.deterministic, profiler=args.profiler,
                         callbacks=[checkpoint_callback])

    data_module = AlignmentDataModule(tokenizer=tokenizer, batch_size=args.batch_size, num_workers=args.num_workers,
                                      alignment_train_path=args.alignment_train_path,
                                      alignment_val_path=args.alignment_val_path, visual_data_path=args.visual_data_dir)

    trainer.fit(alignment_model, datamodule=data_module)


if __name__ == "__main__":
    main()
