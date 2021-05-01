import torch


# From https://stackoverflow.com/a/53403392/1165181
# There's also one in https://github.com/allenai/allennlp/blob/4535f5c/allennlp/nn/util.py#L119
def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = lengths.max()
    return torch.arange(max_length).expand(len(lengths), -1) < lengths.unsqueeze(1)  # noqa
