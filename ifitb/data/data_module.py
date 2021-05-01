from typing import Optional

import pytorch_lightning as pl
from overrides import overrides
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from ifitb.data.fitb_dataset import FitbDataset
from ifitb.data.intention_dataset import IntentionDataset

URL_REASONS_BY_VERB = "https://www.dropbox.com/s/njzm9bes52wzdnm/dict_concept_net_clustered_manual.json?dl=1"
URL_INTENTIONS_DATA = "https://www.dropbox.com/s/93wt5jexgudducu/dict_sentences_per_verb_all_MARKERS.json?dl=1"


class IntentionFitbDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None, batch_size: Optional[int] = 32,
                 eval_batch_size: Optional[int] = None, num_workers: int = 0, t5_format: bool = True,
                 output_visual: bool = True, reasons_by_verb_path: str = URL_REASONS_BY_VERB,
                 intentions_data_path: str = URL_INTENTIONS_DATA) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size

        self.t5_format = t5_format
        self.output_visual = output_visual
        self.reasons_by_verb_path = reasons_by_verb_path
        self.intentions_data_path = intentions_data_path

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = FitbDataset(self.intentions_data_path, self.tokenizer, t5_format=self.t5_format,
                              output_visual=self.output_visual)
        # TODO: bucket-batching could make training faster, and consume less memory.
        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=None if self.eval_batch_size is None else dataset.collate_fn)

    def _eval_dataloader(self) -> DataLoader:
        dataset = IntentionDataset(self.reasons_by_verb_path, self.intentions_data_path, self.tokenizer,
                                   t5_format=self.t5_format, output_visual=self.output_visual)
        # FIXME: divide into val and test
        return DataLoader(dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=None if self.eval_batch_size is None else dataset.collate_fn)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._eval_dataloader()

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader()
