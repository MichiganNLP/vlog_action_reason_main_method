import itertools
import json
import re
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ifitb.data.util import get_video_features
from ifitb.util.file_utils import cached_path
from ifitb.util.mask_utils import get_mask_from_sequence_lengths

RE_WORD_BOUNDARY = re.compile(r"\b")

TYPE_BATCH = MutableMapping[str, Any]


class IntentionDataset(Dataset):
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
        for action, instances in instances_by_action.items():
            for instance in instances:
                verb_end_position = next(RE_WORD_BOUNDARY.finditer(instance["sentence"],
                                                                   instance["verb_pos_sentence"] + 1)).end()
                # TODO: think of a better template?
                text = (f"{instance['sentence_before']} {instance['sentence'][:verb_end_position]}"
                        f" because _____{instance['sentence'][verb_end_position:]} {instance['sentence_after']}")
                self.instances.append({
                    "text": text,
                    "video_id": instance["video"],
                    "video_start_time": instance["time_s"],
                    "video_end_time": instance["time_e"],
                    "verb": action,
                    "choices": instance["reasons"],
                    "ground_truth": instance["answers"],
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

        for k in ["text", "choices", "ground_truth"]:
            stack = batch[k]

            if self.tokenizer:
                counts = None
                if self.t5_format:
                    if k in {"choices", "ground_truth"}:
                        counts = [len(choices) for choices in stack]
                        to_tokenize = [f"<extra_id_0> {c} <extra_id_1>" for choices in stack for c in choices]
                    elif k == "text":
                        to_tokenize = [s.replace("_____", "<extra_id_0>") for s in stack]
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
                if counts:
                    tensor_iter = iter(tokenization_output["input_ids"])
                    tensor_list = [torch.stack(list(itertools.islice(tensor_iter, count))) for count in counts]
                    batch[f"{k}_ids"] = pad_sequence(tensor_list, batch_first=True)
                    # We don't use the attention mask in this case. So we don't compute it.
                else:
                    batch[f"{k}_ids"] = tokenization_output["input_ids"]
                    batch[f"{k}_attention_mask"] = tokenization_output["attention_mask"]

        if "visual" in keys:
            visual_list = batch["visual"]
            batch["visual"] = pad_sequence(visual_list, batch_first=True)

            lengths = torch.as_tensor([visual_instance.size(0) for visual_instance in visual_list])
            batch["visual_attention_mask"] = get_mask_from_sequence_lengths(lengths)

        return batch
