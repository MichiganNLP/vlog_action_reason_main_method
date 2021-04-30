import json
from typing import Any, Mapping, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ifitb.util.file_utils import cached_path


class FitbDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 t5_format: bool = True, output_visual: bool = True) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.t5_format = t5_format
        self.output_visual = output_visual

        with open(cached_path(data_path)) as file:
            instances_by_action = json.load(file)

        self.instances = [
            {
                "text": text,
                "video": instance["video"],
                "start_time": instance["time_s"],
                "end_time": instance["time_e"],
            }
            for instances in instances_by_action.values()
            for instance in instances
            if "because" in (text := f"{instance['sentence_before']} {instance['sentence']}"
                                     f" {instance['sentence_after']}")
        ]

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        # TODO: output visual
        return self.instances[i]

    def __len__(self) -> int:
        return len(self.instances)

    # TODO: collate_fn
