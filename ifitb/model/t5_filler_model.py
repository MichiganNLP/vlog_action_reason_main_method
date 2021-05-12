from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from transformers import AdamW, PreTrainedModel, PreTrainedTokenizerBase, get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput

from ifitb.data.fitb_dataset import TYPE_BATCH as TYPE_FITB_BATCH
from ifitb.data.intention_dataset import TYPE_BATCH as TYPE_INTENTION_BATCH
from ifitb.metrics.metrics import AllMetrics
from ifitb.model.decoding import compute_label_normalized_logits, compute_label_prob


class T5FillerModel(pl.LightningModule):
    """At train time, it fits a single blank. At test time, it determines if each choice is valid or not
    (multi-class)."""

    def __init__(self, t5_like_pretrained_model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                 lr: float = 1e-3, weight_decay: float = 0,  # noqa
                 lr_scheduler: Optional[Literal["linear_with_warmup"]] = "linear_with_warmup") -> None:  # noqa
        super().__init__()

        # These 2 args are too large to serialize.
        self.save_hyperparameters(ignore=["t5_like_pretrained_model", "tokenizer"])

        # The model doesn't necessarily use T5 classes (e.g., `T5PreTrainedModel`).
        # It just needs to be pretrained like T5 and support conditional generation.
        #
        # Commented out as it induced lint warnings in PyCharm 2021.
        # See https://youtrack.jetbrains.com/issue/PY-48654
        # assert isinstance(t5_like_pretrained_model, tuple(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values()))
        self.t5_pretrained_model = t5_like_pretrained_model

        self.tokenizer = tokenizer

        self.threshold = torch.nn.Parameter(torch.tensor(1e-10))

        # TODO: make sure the metric threshold changes when the parameter one does.
        self.all_metrics = AllMetrics(threshold=self.threshold)  # noqa

        self.generate_kwargs = {}

        self.generate_kwargs.setdefault("return_dict_in_generate", True)
        self.generate_kwargs.setdefault("output_scores", True)

        self.extra_id_0 = self.tokenizer.convert_tokens_to_ids(["<extra_id_0>"])[0]
        self.extra_id_1 = self.tokenizer.convert_tokens_to_ids(["<extra_id_1>"])[0]

    @overrides
    def on_epoch_start(self) -> None:
        self.all_metrics.reset()

    @overrides
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                label_ids: Optional[torch.Tensor] = None, revert_changes_to_label_ids: Optional[bool] = None,
                **kwargs) -> Seq2SeqLMOutput:
        # Note that if changes to `label_ids` are reverted the in-place operation it's gonna cause issues with the
        # gradients. So we default to false in training mode and true in eval mode.
        revert_changes_to_label_ids = self.training if revert_changes_to_label_ids is None \
            else revert_changes_to_label_ids

        # Note that passing a mask for the `label_ids` isn't necessary because the decoding is left to right (thus
        # the padding embeddings can only affect the next padding to be generating) and because they are not computed
        # in the loss value if using the `-100` value.

        label_padding_mask = label_ids == self.t5_pretrained_model.config.pad_token_id
        label_ids[label_padding_mask] = -100  # For the loss computation.

        output = self.t5_pretrained_model(input_ids, attention_mask=attention_mask, labels=label_ids, **kwargs)

        if revert_changes_to_label_ids:
            label_ids[label_padding_mask] = self.t5_pretrained_model.config.pad_token_id

        return output

    def training_step(self, batch: TYPE_FITB_BATCH, batch_idx: int = 0) -> torch.Tensor:
        if "text_with_blanks_ids" in batch:  # FITB training
            input_ids = batch.pop("text_with_blanks_ids")
            input_attention_mask = batch.pop("text_with_blanks_attention_mask", None)
            label_ids = batch.pop("label_ids")

            del batch["video_id"]
            del batch["video_start_time"]
            del batch["video_end_time"]
            del batch["text_with_blanks"]
            del batch["label"]
            del batch["label_attention_mask"]
        else:  # Training with labeled data.
            input_ids = batch.pop("text_ids")
            input_attention_mask = batch.pop("text_attention_mask", None)

            ground_truth_ids = batch.pop("ground_truth_ids")
            selected_ground_truth_id_list = []
            for ground_truth_ids_instance in ground_truth_ids:
                # Randomly select one ground truth option, supposing there's one.
                ground_truth_size = sum(1 for g in ground_truth_ids_instance if g.any())
                i = torch.randint(ground_truth_size, ()).item()
                selected_ground_truth_id_list.append(ground_truth_ids_instance[i])
            label_ids = torch.stack(selected_ground_truth_id_list)

            del batch["text"]
            del batch["video_id"]
            del batch["video_start_time"]
            del batch["video_end_time"]
            del batch["verb"]
            del batch["choices"]
            del batch["choices_ids"]
            del batch["ground_truth"]

        loss = self(input_ids, input_attention_mask, label_ids, revert_changes_to_label_ids=False, **batch)["loss"]
        self.log("loss", loss)
        return loss

    def _eval_step(self, text_ids: torch.Tensor, text_attention_mask: torch.Tensor,
                   choices_ids: torch.Tensor, text: Sequence[str], verb: Sequence[str],
                   choices: Sequence[Sequence[str]], ground_truth: Sequence[Sequence[str]],
                   video_id: Optional[Sequence[str]], video_start_time: Optional[Sequence[str]],
                   video_end_time: Optional[Sequence[str]], log_prefix: str = "", **kwargs) -> None:
        self.write_prediction("video_id", video_id)  # noqa
        self.write_prediction("video_start_time", video_start_time)  # noqa
        self.write_prediction("video_end_time", video_end_time)  # noqa
        self.write_prediction("text", text)  # noqa
        self.write_prediction("choices", choices)  # noqa

        del kwargs["ground_truth_ids"]

        # Encode the input only once, then reuse for the choices.
        kwargs = self.t5_pretrained_model._prepare_encoder_decoder_kwargs_for_generation(
            text_ids,  # noqa
            {"attention_mask": text_attention_mask, "output_hidden_states": True, "output_attentions": True, **kwargs})

        # TODO (optimization): `expand` every variable, reshape choice_ids
        # batch_size, max_choice_count = choices_ids.shape[:2]
        #
        # text_ids = (text_ids
        #             .unsqueeze(1)
        #             .expand(-1, max_choice_count, -1)
        #             .view(batch_size * max_choice_count, -1))
        # text_attention_mask = (text_attention_mask
        #                        .unsqueeze(1)
        #                        .expand(-1, max_choice_count, -1)
        #                        .view(batch_size * max_choice_count, -1))

        float_dtype = kwargs["encoder_outputs"].last_hidden_state.dtype

        choices_prob = torch.empty_like(choices_ids[:, :, 0], dtype=float_dtype)

        for i, choice_ids in enumerate(choices_ids.transpose(1, 0)):
            # `clone()` so `view()` works when computing the cross-entropy loss.
            # It's also convenient so we skip reverting the values as well.
            output = self(text_ids, label_ids=choice_ids.clone(), revert_changes_to_label_ids=False, **kwargs)
            self.log(f"{log_prefix}loss", output.loss, prog_bar=True)

            choice_normalized_logits = compute_label_normalized_logits(output.logits, choice_ids,
                                                                       self.t5_pretrained_model.config,
                                                                       ignore_eos_token=True)
            choices_prob[:, i] = compute_label_prob(choice_normalized_logits)

        # Use a list to remove the padding but also because using tensor will cause issues. It's because they have
        # different sizes when concatenating the tensors across batches, depending on the max number of choices in each
        # batch.
        choices_prob_list = [[choice_prob.item()
                              for choice_prob, choice_ids in zip(choices_prob_instance, choices_ids_instance)
                              if (choice_ids != 0).any()]
                             for choices_prob_instance, choices_ids_instance in zip(choices_prob, choices_ids)]
        self.write_prediction("choices_prob", choices_prob_list)  # noqa

        # generated_output = self.t5_pretrained_model.generate(text_ids, **self.generate_kwargs, **kwargs)  # noqa
        #
        # generated_ids = generated_output.sequences
        # generated = self.tokenizer.batch_decode(
        #     compute_first_blank(generated_ids, self.t5_pretrained_model.config.decoder_start_token_id,
        #                         self.extra_id_0, self.extra_id_1))

        for k, v in self.all_metrics(choices_prob_list, ground_truth, verb, choices,
                                     device=next(self.parameters()).device).items():  # noqa
            if k in {"accuracy", "f1", "ground_truth_prob", "perplexity"}:
                self.log(f"{log_prefix}{k}_step", v, prog_bar=True)

        predicted = [[choice for choice, prob in zip(verb_choices, verb_choices_prob_list) if prob > self.threshold]
                     for verb_choices, verb_choices_prob_list in zip(choices, choices_prob_list)]
        self.write_prediction("predicted", predicted)  # noqa

    @overrides
    def validation_step(self, batch: TYPE_INTENTION_BATCH, batch_idx: int = 0) -> None:
        self._eval_step(**batch, log_prefix="val_")

    @overrides
    def test_step(self, batch: TYPE_INTENTION_BATCH, batch_idx: int = 0) -> None:
        self._eval_step(**batch, log_prefix="test_")

    def _on_epoch_end(self, log_prefix: str = "") -> None:
        for k, v in self.all_metrics.compute().items():
            self.log(f"{log_prefix}{k}", v, prog_bar=k in {"accuracy", "f1", "ground_truth_prob"})

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
