import json
import numbers
from pathlib import Path
from typing import Any, Iterable, Iterator, MutableMapping, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from ifitb.util.file_utils import cached_path
from ifitb.util.mask_utils import get_mask_from_sequence_lengths

TYPE_BATCH = MutableMapping[str, Any]


class TextVisualAlignmentDataset(IterableDataset):  # noqa
    def __init__(self, vatex_data_path: str, visual_data_path: str,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        super().__init__()

        self.visual_data_path = Path(cached_path(visual_data_path))
        self.tokenizer = tokenizer

        with open(cached_path(vatex_data_path)) as file:
            self.instances = json.load(file)

    def __iter__(self) -> Iterator[TYPE_BATCH]:
        n = len(self.instances)
        while True:
            i = torch.randint(n, size=()).item()
            instance = self.instances[i]
            captions = instance["enCap"]

            output = {
                "text": captions[torch.randint(len(captions), size=()).item()],
                "label": torch.rand(1).item() < .5,
            }

            weights = torch.full((n,), 1 / (n - 1))
            weights[i] = 0
            video_index = i if output["label"] else torch.multinomial(weights, num_samples=1).item()
            video_id = self.instances[video_index]["videoID"]
            output["visual"] = torch.from_numpy(np.load(self.visual_data_path / f"{video_id}.npy")).squeeze(0)  # noqa

            yield output

    def collate_fn(self, instances: Iterable[TYPE_BATCH]) -> TYPE_BATCH:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        for k in {"text"}:
            stack = batch[k]

            if self.tokenizer:
                to_tokenize = stack

                # We tokenize in batches, in parallel. Probably there's a little gain than each worker tokenizing
                # separately each item in a batch because the padding is known a priori and there may be other parallel
                # optimizations. And it's more elegant. Still, it's likely marginal. Though now the workers aren't
                # serial anymore, so we shouldn't use as many workers as CPU cores but just a small number so the
                # compute devices aren't starving but not large so they never compete a lot with each other (esp. at the
                # beginning, where the pipeline of workers is starting).
                tokenization_output = self.tokenizer(to_tokenize, padding="longest", truncation=True,
                                                     return_tensors="pt")
                batch[f"{k}_ids"] = tokenization_output["input_ids"]
                batch[f"{k}_attention_mask"] = tokenization_output["attention_mask"]

        for k in keys:
            stack = batch[k]
            if isinstance(batch[k], (list, tuple)) and isinstance(stack[0], numbers.Number):
                batch[k] = torch.tensor(stack)

        visual_list = batch["visual"]
        batch["visual"] = pad_sequence(visual_list, batch_first=True)

        lengths = torch.as_tensor([visual_instance.size(0) for visual_instance in visual_list])
        batch["visual_attention_mask"] = get_mask_from_sequence_lengths(lengths)

        return batch
