import torch
from transformers import PretrainedConfig


def compute_label_normalized_logits(logits: torch.Tensor, label_ids: torch.Tensor, model_config: PretrainedConfig,
                                    ignore_eos_token: bool = False) -> torch.Tensor:
    """Computes the normalized logits of the given label token IDs.

    `logits` has shape (N, L, V) and dtype float.
    `label_ids` has shape (N, L) and dtype int.

    Returned tensor has shape (N, L) and dtype float.
    """
    if model_config.decoder_start_token_id is not None \
            and (label_ids[:, 0] == model_config.decoder_start_token_id).all():  # noqa
        label_ids = label_ids[:, 1:]

    N, L = label_ids.shape

    normalized_logits = logits.log_softmax(dim=-1)  # Normalize so we can later compute the probability.

    label_normalized_logits = normalized_logits[torch.arange(N)[:, None], torch.arange(L)[None], label_ids]

    if model_config.pad_token_id is not None:
        label_normalized_logits[label_ids == model_config.pad_token_id] = 0

    if ignore_eos_token and model_config.eos_token_id is not None:
        label_normalized_logits[label_ids == model_config.eos_token_id] = 0

    return label_normalized_logits


def compute_label_prob(label_normalized_logits: torch.Tensor) -> torch.Tensor:
    """Computes the joint probability of the given label.

    `label_normalized_logits` has shape (N, L) and dtype float.

    Returned tensor has shape (N,) and dtype float.
    """
    return label_normalized_logits.sum(dim=-1).exp()
