from typing import Optional

import pytorch_lightning as pl
from overrides import overrides
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from ifitb.data.fitb_dataset import FitbDataset
from ifitb.data.intention_dataset import IntentionDataset

URL_FITB_DATA = "https://www.dropbox.com/s/0wyll5tvdjzu7zu/" \
                "dict_sentences_per_verb_all_MARKERS_without_val_and_test.json?dl=1"

URL_INTENTIONS_TRAIN = "https://www.dropbox.com/s/0zrvjcq90sxi6e4/train2.json?dl=1"
URL_INTENTIONS_VAL = "https://www.dropbox.com/s/zfdpv6h4rcwa8bm/val2.json?dl=1"
URL_INTENTIONS_TEST = "https://www.dropbox.com/s/mce8szi50tdpvwv/test2.json?dl=1"

URL_VISUAL_FEATURES = "https://www.dropbox.com/s/k4zjwcdz4lksv0j/i3d_video_features.tar.gz?dl=1!i3d_video_features"


class IntentionFitbDataModule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None, batch_size: Optional[int] = 32,
                 eval_batch_size: Optional[int] = None, num_workers: int = 0, t5_format: bool = True,
                 output_visual: bool = True, fitb_data_path: str = URL_FITB_DATA,
                 intentions_train_path: str = URL_INTENTIONS_TRAIN, intentions_val_path: str = URL_INTENTIONS_VAL,
                 intentions_test_path: str = URL_INTENTIONS_TEST, visual_data_path: str = URL_VISUAL_FEATURES) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size

        self.t5_format = t5_format
        self.output_visual = output_visual

        self.fitb_data_path = fitb_data_path

        self.intentions_train_path = intentions_train_path
        self.intentions_val_path = intentions_val_path
        self.intentions_test_path = intentions_test_path

        self.visual_data_path = visual_data_path

    def fitb_dataloader(self) -> DataLoader:
        dataset = FitbDataset(self.fitb_data_path, self.tokenizer, t5_format=self.t5_format,
                              output_visual=self.output_visual, visual_data_path=self.visual_data_path)
        # TODO: bucket-batching could make training faster, and consume less memory.
        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=None if self.eval_batch_size is None else dataset.collate_fn,
                          persistent_workers=self.num_workers > 0)

    def _dataloader(self, data_path: str, batch_size: int, train: bool) -> DataLoader:
        dataset = IntentionDataset(data_path, self.tokenizer, t5_format=self.t5_format,
                                   output_visual=self.output_visual, visual_data_path=self.visual_data_path)
        return DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=self.num_workers,
                          pin_memory=True, collate_fn=None if batch_size is None else dataset.collate_fn,
                          persistent_workers=self.num_workers > 0)

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.intentions_train_path, batch_size=self.batch_size, train=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.intentions_val_path, batch_size=self.eval_batch_size, train=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.intentions_test_path, batch_size=self.eval_batch_size, train=False)
