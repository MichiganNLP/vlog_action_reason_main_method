from collections import defaultdict
from typing import Iterator, MutableMapping, Sequence

import torch
from overrides import overrides
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics import Accuracy, Metric


class AccuracyPerAction(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.binarizer_by_verb: MutableMapping[str, MultiLabelBinarizer] = {}
        self.metric_by_verb = defaultdict(lambda: Accuracy())

    @overrides
    def update(self, preds: Iterator[Sequence[str]], targets: Iterator[Sequence[str]], verbs: Iterator[str],
               choices: Iterator[Sequence[str]]) -> None:
        for pred, target, verb, verb_choices in zip(preds, targets, verbs, choices):
            if binarizer := self.binarizer_by_verb.get(verb):
                assert binarizer.classes == verb_choices, (f"The intention choices for an instance of the verb '{verb}'"
                                                           f" are different than a previous instance of the same verb.")
            else:
                # `fit` needs to be called.
                binarizer = self.binarizer_by_verb[verb] = MultiLabelBinarizer(classes=verb_choices).fit([])

            encoded_pred, encoded_target = torch.from_numpy(binarizer.transform([pred, target]))

            self.metric_by_verb[verb].update(encoded_pred, encoded_target)

    @overrides
    def compute(self) -> torch.Tensor:
        return torch.stack([metric.compute() for metric in self.metric_by_verb.values()]).mean()
