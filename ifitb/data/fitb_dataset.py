import itertools
import json
import re
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ifitb.data.util import get_video_features, video_feature_file_exists
from ifitb.util.file_utils import cached_path
from ifitb.util.mask_utils import get_mask_from_sequence_lengths

RE_BLANK = re.compile(r"_____")

TYPE_BATCH = MutableMapping[str, Any]


def _blank_reason(text: str) -> Tuple[str, Optional[str]]:
    keyword = "because"
    if (start := text.find(keyword)) == -1 or (end := start + len(keyword)) + 1 >= len(text):
        return text, None
    else:
        return f"{text[:end]} _____", text[end + 1:]


def _format_blanks_for_t5(text: str) -> str:
    count_iter = itertools.count()
    return RE_BLANK.sub(lambda x: f"<extra_id_{next(count_iter)}>", text)


class FitbDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 t5_format: bool = True, output_visual: bool = True, visual_data_path: Optional[str] = None) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.t5_format = t5_format
        self.output_visual = output_visual
        self.visual_data_path = cached_path(visual_data_path) if visual_data_path else None

        with open(cached_path(data_path)) as file:
            instances_by_action = json.load(file)

        self.instances = []
        for instances in instances_by_action.values():
            for instance in instances:
                text_with_blanks, labels = zip(_blank_reason(instance["sentence_before"]),
                                               _blank_reason(instance["sentence"]),
                                               _blank_reason(instance["sentence_after"]))

                video_id = instance["video"]
                video_start_time = instance["time_s"]
                video_end_time = instance["time_e"]

                if (labels := [label for label in labels if label]) \
                        and (not self.output_visual or video_feature_file_exists(self.visual_data_path, video_id,
                                                                                 video_start_time, video_end_time)):
                    self.instances.append({
                        "text_with_blanks": " ".join(text_with_blanks),
                        "label": labels,
                        "video_id": video_id,
                        "video_start_time": video_start_time,
                        "video_end_time": video_end_time,
                    })

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        instance = self.instances[i]

        if self.output_visual and "visual" not in instance:
            instance["visual"] = get_video_features(self.visual_data_path, instance["video_id"],
                                                    instance["video_start_time"], instance["video_end_time"])

        return instance

    def __len__(self) -> int:
        return len(self.instances)

    # noinspection DuplicatedCode
    def collate_fn(self, instances: Iterable[TYPE_BATCH]) -> TYPE_BATCH:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        for k in ["text_with_blanks", "label"]:
            stack = batch[k]

            if self.tokenizer:
                if self.t5_format:
                    if k == "label":
                        to_tokenize = [" ".join(f"<extra_id_{i}> {label} <extra_id_{i + 1}>"
                                                for i, label in enumerate(labels_instance))
                                       for labels_instance in stack]
                    elif k == "text_with_blanks":
                        to_tokenize = [_format_blanks_for_t5(s) for s in stack]
                    else:
                        to_tokenize = stack
                else:
                    to_tokenize = stack

                # We tokenize in batches, in parallel. Probably there's a little gain than each worker tokenizing
                # separately each item in a batch because the padding is known a priori and there may be other parallel
                # optimizations. And it's more elegant. Still, it's likely marginal. Though now the workers aren't
                # serial anymore, so we shouldn't use as many workers as CPU cores but just a small number so the
                # devices aren't starving but not large so they never compete a lot with each other (esp. at the
                # beginning, where the pipeline of workers is starting).
                tokenization_output = self.tokenizer(to_tokenize, padding="longest", truncation=True,
                                                     return_tensors="pt")
                batch[f"{k}_ids"] = tokenization_output["input_ids"]
                batch[f"{k}_attention_mask"] = tokenization_output["attention_mask"]

        if "visual" in keys:
            visual_list = batch["visual"]
            batch["visual"] = pad_sequence(visual_list, batch_first=True)

            lengths = torch.as_tensor([visual_instance.size(0) for visual_instance in visual_list])
            batch["visual_attention_mask"] = get_mask_from_sequence_lengths(lengths)

        return batch
