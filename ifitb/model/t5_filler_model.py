import inspect
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from overrides import overrides
from pytorch_lightning.utilities.parsing import get_init_args
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from transformers import AdamW, PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from ifitb.data.intention_dataset import TYPE_BATCH as TYPE_INTENTION_BATCH
from ifitb.model.decoding import compute_label_normalized_logits, compute_label_prob


class T5FillerModel(pl.LightningModule):
    """At train time, it fits a single blank. At test time, it determines if each choice is valid or not
    (multi-class)."""

    def __init__(self, t5_like_pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                 lr: float = 1e-3, weight_decay: float = 0,  # noqa
                 lr_scheduler: Optional[Literal["linear_with_warmup"]] = "linear_with_warmup") -> None:  # noqa
        super().__init__()

        frame = inspect.currentframe()
        init_args = get_init_args(frame)
        # The following 2 are too large to serialize:
        del init_args["t5_like_pretrained_model"]
        del init_args["tokenizer"]
        self.save_hyperparameters(init_args)

        # The model doesn't necessarily use T5 classes (e.g., `T5PreTrainedModel`).
        # It just needs to be pretrained like T5 and support conditional generation.
        # Commented out as it induced lint warnings in PyCharm 2021.
        # assert isinstance(t5_like_pretrained_model, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()))
        self.t5_pretrained_model = t5_like_pretrained_model

        self.tokenizer = tokenizer
        # self.metrics = ...  # TODO

        self.extra_id_0 = self.tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        self.extra_id_1 = self.tokenizer.convert_tokens_to_ids(["<extra_id_1>"])[0]

        self.all_token_ids = torch.arange(self.t5_pretrained_model.config.vocab_size)  # noqa

    @overrides
    def on_epoch_start(self) -> None:
        # self.all_metrics.reset()
        pass

    @overrides
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                label_ids: Optional[torch.Tensor] = None, revert_changes_to_label_ids: bool = True,
                **kwargs) -> Seq2SeqLMOutput:
        # Note that passing a mask for the `label_ids` isn't necessary because the decoding is left to right (thus
        # the padding embeddings can only affect the next padding to be generating) and because they are not computed
        # in the loss value if using the `-100` value.

        label_padding_mask = label_ids == self.t5_pretrained_model.config.pad_token_id
        label_ids[label_padding_mask] = -100  # For the loss computation.

        output = self.t5_pretrained_model(input_ids, attention_mask=attention_mask, labels=label_ids, **kwargs)

        if revert_changes_to_label_ids:
            label_ids[label_padding_mask] = self.t5_pretrained_model.config.pad_token_id

        return output

    # def training_step(self, batch: TYPE_BATCH, batch_idx: int = 0) -> torch.Tensor:  # TODO
    #     text_ids = batch.pop("text_ids")
    #     text_attention_mask = batch.pop("text_attention_mask", None)
    #     choices_ids = batch.pop("choices_ids")
    #
    #     del batch["video_id"]
    #     del batch["video_start_time"]
    #     del batch["video_end_time"]
    #     del batch["text"]
    #     del batch["choices"]
    #
    #     loss = self(text_ids, text_attention_mask, choices_ids, **batch)["loss"]
    #     self.log("loss", loss)
    #     return loss

    def _eval_step(self, text_ids: torch.Tensor, text_attention_mask: torch.Tensor,
                   choices_ids: torch.Tensor, text: Sequence[str], choices: Sequence[Sequence[str]],
                   video_id: Optional[Sequence[str]], video_start_time: Optional[Sequence[str]],
                   video_end_time: Optional[Sequence[str]], log_prefix: str = "", **kwargs) -> None:
        self.write_prediction("video_id", video_id)  # noqa
        self.write_prediction("video_start_time", video_start_time)  # noqa
        self.write_prediction("video_end_time", video_end_time)  # noqa
        self.write_prediction("text", text)  # noqa
        self.write_prediction("choices", choices)  # noqa

        # Compute the first choice of each instance, save the encoder output, then compute the rest.

        first_choice_ids = choices_ids[:, 0].clone()  # `clone` so `view` works when computing the cross-entropy loss.

        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("output_attentions", True)

        output = self(text_ids, text_attention_mask, first_choice_ids, **kwargs)
        self.log(f"{log_prefix}loss", output.loss, prog_bar=True)

        choices_normalized_logits = compute_label_normalized_logits(output.logits, first_choice_ids,
                                                                    self.t5_pretrained_model.config,
                                                                    ignore_eos_token=True)
        choices_prob = compute_label_prob(choices_normalized_logits)
        self.write_prediction("choices_prob", choices_prob)

        if choices_ids.shape[1] > 1:
            if self.t5_pretrained_model.config.is_encoder_decoder:
                # Reuse the encoder output to avoid computing it multiple times.
                kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=output.encoder_hidden_states[-1],
                                                            hidden_states=output.encoder_hidden_states,
                                                            attentions=output.encoder_attentions)

            # TODO: compute it for all the choices.

        # perplexity_mask = ((choices_ids != self.t5_pretrained_model.config.pad_token_id)
        #                    & (choices_ids != self.t5_pretrained_model.config.eos_token_id))

        # for k, v in self.all_metrics(video_id, label, additional_answers, generated, label_prob,  # TODO
        #                              label_probs, perplexity_mask).items():  # noqa
        #     if k in {"accuracy", "f1_score", "ground_truth_prob", "perplexity"}:
        #         self.log(f"{log_prefix}{k}_step", v, prog_bar=True)

    @overrides
    def validation_step(self, batch: TYPE_INTENTION_BATCH, batch_idx: int = 0) -> None:
        self._eval_step(**batch, log_prefix="val_")

    @overrides
    def test_step(self, batch: TYPE_INTENTION_BATCH, batch_idx: int = 0) -> None:
        self._eval_step(**batch, log_prefix="test_")

    def _on_epoch_end(self, log_prefix: str = "") -> None:
        # for k, v in self.all_metrics.compute().items():  # TODO
        #     self.log(f"{log_prefix}{k}", v, prog_bar=k in {"accuracy", "f1_score", "ground_truth_prob"})
        pass

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end(log_prefix="val_")

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end(log_prefix="test_")

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
