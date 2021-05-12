from typing import Optional

import pytorch_lightning as pl
from overrides import overrides
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from ifitb.data.fitb_dataset import FitbDataset
from ifitb.data.intention_dataset import IntentionDataset

URL_FITB_DATA = "https://www.dropbox.com/s/93wt5jexgudducu/dict_sentences_per_verb_all_MARKERS.json?dl=1"
# TODO: make sure we don't use test data in FITB.

URL_INTENTIONS_TRAIN = "https://www.dropbox.com/s/tqjtsp72ut2v9rd/dict_web_trial_train_santi.json?dl=1"
URL_INTENTIONS_TEST = "https://www.dropbox.com/s/4h78b71294r6fws/dict_web_trial_test_santi.json?dl=1"

URL_VISUAL_FEATURES = "https://www.dropbox.com/s/k4zjwcdz4lksv0j/i3d_video_features.tar.gz?dl=1!i3d_video_features"


class IntentionFitbDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None, batch_size: Optional[int] = 32,
                 eval_batch_size: Optional[int] = None, num_workers: int = 0, t5_format: bool = True,
                 output_visual: bool = True, fitb_data_path: str = URL_FITB_DATA,
                 intentions_train_path: str = URL_INTENTIONS_TRAIN,
                 intentions_test_path: str = URL_INTENTIONS_TEST, visual_data_path: str = URL_VISUAL_FEATURES) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size

        self.t5_format = t5_format
        self.output_visual = output_visual

        self.fitb_data_path = fitb_data_path

        self.intentions_train_path = intentions_train_path  # Unused for now.
        self.intentions_test_path = intentions_test_path

        self.visual_data_path = visual_data_path

    @overrides
    def train_dataloader(self) -> DataLoader:
        dataset = FitbDataset(self.fitb_data_path, self.tokenizer, t5_format=self.t5_format,
                              output_visual=self.output_visual, visual_data_path=self.visual_data_path)
        # TODO: bucket-batching could make training faster, and consume less memory.
        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=None if self.eval_batch_size is None else dataset.collate_fn)

    def _eval_dataloader(self) -> DataLoader:
        dataset = IntentionDataset(self.intentions_test_path, self.tokenizer, t5_format=self.t5_format,
                                   output_visual=self.output_visual, visual_data_path=self.visual_data_path)
        # FIXME: divide into val and test
        return DataLoader(dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, pin_memory=True,
                          collate_fn=None if self.eval_batch_size is None else dataset.collate_fn)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._eval_dataloader()

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader()
