from typing import Optional

import pytorch_lightning as pl
from overrides import overrides
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from ifitb.data.alignment_dataset import TextVisualAlignmentDataset

ALIGNMENT_TRAIN_PATH = "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json"
ALIGNMENT_VAL_PATH = "https://eric-xw.github.io/vatex-website/data/vatex_validation_v1.0.json"
VISUAL_DATA_PATH = "https://vatex-feats.s3.amazonaws.com/trainval.zip!val"


class AlignmentDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None, batch_size: Optional[int] = 32,
                 eval_batch_size: Optional[int] = None, num_workers: int = 0, t5_format: bool = True,
                 alignment_train_path: str = ALIGNMENT_TRAIN_PATH, alignment_val_path: str = ALIGNMENT_VAL_PATH,
                 visual_data_path: str = VISUAL_DATA_PATH) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size

        self.t5_format = t5_format

        self.alignment_train_path = alignment_train_path
        self.alignment_val_path = alignment_val_path

        self.visual_data_path = visual_data_path

    def _dataloader(self, data_path: str, batch_size: int) -> DataLoader:
        dataset = TextVisualAlignmentDataset(data_path, visual_data_path=self.visual_data_path,
                                             tokenizer=self.tokenizer)
        # TODO: bucket-batching could make training faster, and consume less memory.
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=None if batch_size is None else dataset.collate_fn)

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.alignment_train_path, batch_size=self.batch_size)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.alignment_val_path, batch_size=self.eval_batch_size)
