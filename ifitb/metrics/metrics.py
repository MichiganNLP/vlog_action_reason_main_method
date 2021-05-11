from collections import defaultdict
from typing import Iterator, Mapping, MutableMapping, Sequence

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


class AllMetrics(Metric):
    def __init__(self) -> None:  # , compute_prob: bool = True
        super().__init__()
        self.metrics: MutableMapping[str, Metric] = {
            "accuracy": AccuracyPerAction(),
        }

        # if compute_prob:
        #     self.metrics["ground_truth_prob"] = Average()
        #     self.metrics["perplexity"] = Perplexity()

    @overrides
    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    # def __call__(self, video_ids: Sequence[str], label_prob: Optional[torch.Tensor] = None,
    #              label_probs: Optional[torch.Tensor] = None,
    #              perplexity_mask: Optional[torch.Tensor] = None) -> Mapping[str, torch.Tensor]:
    @overrides
    def update(self, preds: Iterator[Sequence[str]], targets: Iterator[Sequence[str]], verbs: Iterator[str],
               choices: Iterator[Sequence[str]]) -> Mapping[str, torch.Tensor]:
        output = {
            "accuracy": self.metrics["accuracy"](preds, targets, verbs, choices),
        }

        # if ground_truth_prob_metric := self.metrics.get("ground_truth_prob"):
        #     assert label_prob is not None
        #     output["ground_truth_prob"] = ground_truth_prob_metric(label_prob)
        #
        # if perplexity_metric := self.metrics.get("perplexity"):
        #     assert label_probs is not None and perplexity_mask is not None
        #     output["perplexity"] = perplexity_metric(label_probs, perplexity_mask)

        return output

    @overrides
    def compute(self) -> Mapping[str, torch.Tensor]:
        return {name: metrics.compute() for name, metrics in self.metrics.items()}
