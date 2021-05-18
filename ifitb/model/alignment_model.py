from typing import Iterable, Literal, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from torchmetrics import Accuracy
from transformers import AdamW, PreTrainedModel, get_linear_schedule_with_warmup

from ifitb.data.alignment_dataset import TYPE_BATCH
from ifitb.util.mask_utils import mean


class AlignmentModel(pl.LightningModule):
    def __init__(self, model: PreTrainedModel, lr: float = 1e-3, weight_decay: float = 0,  # noqa
                 lr_scheduler: Optional[Literal["linear_with_warmup"]] = "linear_with_warmup") -> None:  # noqa
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # The arg is too large to serialize.
        self.model = model

        # self.linear1 = nn.Linear(2 * self.model.config.hidden_size, 100)  # noqa
        # self.activation1 = nn.ReLU()
        # self.linear2 = nn.Linear(100, 1)

        self.loss = nn.BCELoss()

        self.accuracy = Accuracy()

    @overrides
    def on_epoch_start(self) -> None:
        self.accuracy.reset()

    @overrides
    def forward(self, text_ids: torch.Tensor, visual: torch.Tensor, text_attention_mask: Optional[torch.Tensor] = None,
                visual_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # return self.model.encoder(input_ids, attention_mask=attention_mask, visual=visual,
        #                           visual_attention_mask=visual_attention_mask).last_hidden_state
        return self.model.encoder.embed_tokens(text_ids), self.model.encoder.embed_video(visual)

    def _step(self, batch: TYPE_BATCH, log_prefix: str = "") -> torch.Tensor:
        del batch["text"]

        text_ids = batch.pop("text_ids")
        text_attention_mask = batch.pop("text_attention_mask", None)
        visual = batch.pop("visual")
        visual_attention_mask = batch.pop("visual_attention_mask", None)
        label = batch.pop("label")

        encoded = self(text_ids, visual=visual, text_attention_mask=text_attention_mask,
                       visual_attention_mask=visual_attention_mask)

        # max_text_length = text_ids.shape[-1]
        # encoded_text = mean(encoded[:, :max_text_length], dim=1, mask=text_attention_mask)
        # encoded_visual = mean(encoded[:, max_text_length:], dim=1, mask=visual_attention_mask)
        encoded_text = mean(encoded[0], dim=1, mask=text_attention_mask)
        encoded_visual = mean(encoded[1], dim=1, mask=visual_attention_mask)
        # encoded_text, encoded_visual = encoded

        scores = (encoded_text * encoded_visual).sum(-1)
        # scores = self.linear2(self.activation1(self.linear1(torch.cat((encoded_text, encoded_visual),
        #                                                               dim=1)))).squeeze(-1)

        # scores = torch.matmul(encoded_text, encoded_visual.transpose(1, 2)).max((1, 2))
        # scores = F.cosine_similarity(encoded_text.unsqueeze(2), encoded_visual.unsqueeze(1), dim=3).amax((1, 2))
        # probs = scores

        probs = torch.sigmoid(scores)

        loss = self.loss(probs, label.to(visual.dtype))

        accuracy = self.accuracy(probs, label)

        if not self.trainer.sanity_checking:
            self.log(f"{log_prefix}loss", loss)
            self.log(f"{log_prefix}accuracy", accuracy, prog_bar=True)

        return loss

    @overrides
    def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:
        return self._step(batch, log_prefix="train_")

    @overrides
    def validation_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> None:
        self._step(batch, log_prefix="val_")

    @overrides
    def test_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> None:
        self._step(batch, log_prefix="test_")

    @overrides
    def configure_optimizers(self) -> Union[Iterable[Optimizer], Tuple[Iterable[Optimizer], Iterable[_LRScheduler]]]:
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.lr_scheduler and (epochs := self.trainer.max_epochs):
            if self.hparams.lr_scheduler == "linear_with_warmup":
                scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * epochs, epochs)  # noqa
            else:
                raise ValueError(f"Unrecognized LR Scheduler '{self.hparams.lr_scheduler}'")

            return [optimizer], [scheduler]
        else:
            return [optimizer]
