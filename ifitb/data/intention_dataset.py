import itertools
import json
import re
from typing import Any, Iterable, Mapping, MutableMapping, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ifitb.util.file_utils import cached_path
from ifitb.util.mask_utils import get_mask_from_sequence_lengths

RE_WORD_BOUNDARY = re.compile(r"\b")

TYPE_BATCH = MutableMapping[str, Any]


class IntentionDataset(Dataset):
    def __init__(self, reasons_by_verb_path: str, data_path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 t5_format: bool = True, output_visual: bool = True) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.t5_format = t5_format
        self.output_visual = output_visual

        with open(cached_path(reasons_by_verb_path)) as file:
            choices_by_action = json.load(file)

        with open(cached_path(data_path)) as file:
            instances_by_action = json.load(file)

        self.instances = []
        for action, instances in instances_by_action.items():
            for instance in instances:
                if choices := choices_by_action.get(action):  # FIXME: there are some missing verbs (e.g., "accomplish")
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
                        "choices": choices,
                        # TODO: gt
                    })

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        # TODO: output visual
        return self.instances[i]

    def __len__(self) -> int:
        return len(self.instances)

    # noinspection DuplicatedCode
    def collate_fn(self, instances: Iterable[TYPE_BATCH]) -> TYPE_BATCH:
        keys = next(iter(instances), {})
        batch = {k: [instance[k] for instance in instances] for k in keys}

        for k in ["text", "choices"]:
            stack = batch[k]

            if self.tokenizer:
                counts = None
                if self.t5_format:
                    if k == "choices":
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
                # compute devices aren't starving but not large so they never compete a lot with each other (esp. at the
                # beginning, where the pipeline of workers is starting).
                tokenization_output = self.tokenizer(to_tokenize, padding="longest", truncation=True,
                                                     return_tensors="pt")
                if counts:
                    tensor_iter = iter(tokenization_output["input_ids"])
                    tensor_list = [torch.stack(list(itertools.islice(tensor_iter, count))) for count in counts]
                    batch[f"{k}_ids"] = pad_sequence(tensor_list, batch_first=True)
                    # We don't use the attention mask for the choices, so we don't compute it.
                else:
                    batch[f"{k}_ids"] = tokenization_output["input_ids"]
                    batch[f"{k}_attention_mask"] = tokenization_output["attention_mask"]

        if "visual" in keys:
            visual_list = batch["visual"]
            batch["visual"] = pad_sequence(visual_list, batch_first=True)

            lengths = torch.as_tensor([visual_instance.size(0) for visual_instance in visual_list])
            batch["visual_attention_mask"] = get_mask_from_sequence_lengths(lengths)

        return batch
