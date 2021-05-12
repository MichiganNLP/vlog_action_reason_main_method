from collections import defaultdict
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Optional, Sequence

import torch
from overrides import overrides
from sklearn.preprocessing import MultiLabelBinarizer
from torchmetrics import Accuracy, F1, Metric, Precision, Recall


class MetricPerAction(Metric):
    def __init__(self, metric_class: Callable[[], Metric]) -> None:
        super().__init__()
        self.binarizer_by_verb: MutableMapping[str, MultiLabelBinarizer] = {}
        self.metric_by_verb = defaultdict(metric_class)

    @overrides
    def update(self, preds: Iterator[Sequence[float]], targets: Iterator[Sequence[str]], verbs: Iterator[str],
               choices: Iterator[Sequence[str]], device: Optional[Any] = None) -> None:
        for pred, target, verb, verb_choices in zip(preds, targets, verbs, choices):
            if binarizer := self.binarizer_by_verb.get(verb):
                assert binarizer.classes == verb_choices, (f"The intention choices for an instance of the verb '{verb}'"
                                                           f" are different than a previous instance of the same verb.")
            else:
                # `fit` needs to be called.
                binarizer = self.binarizer_by_verb[verb] = MultiLabelBinarizer(classes=verb_choices).fit([])

            pred = torch.tensor(pred).unsqueeze(0)
            encoded_target = torch.from_numpy(binarizer.transform([target]))

            if device is not None:
                pred = pred.to(device)
                encoded_target = encoded_target.to(device)

            self.metric_by_verb[verb].update(pred, encoded_target)

    @overrides
    def compute(self) -> torch.Tensor:
        return torch.stack([metric.compute() for metric in self.metric_by_verb.values()]).mean()


class AllMetrics(Metric):
    def __init__(self, threshold: float = 0.5) -> None:  # , compute_prob: bool = True
        super().__init__()
        self.metrics: MutableMapping[str, Metric] = {
            "accuracy": MetricPerAction(lambda: Accuracy(threshold=threshold, average="samples")),
            "f1": MetricPerAction(lambda: F1(threshold=threshold, average="samples")),
            "precision": MetricPerAction(lambda: Precision(threshold=threshold, average="samples")),
            "recall": MetricPerAction(lambda: Recall(threshold=threshold, average="samples")),
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
    def update(self, preds: Iterator[Sequence[float]], targets: Iterator[Sequence[str]], verbs: Iterator[str],
               choices: Iterator[Sequence[str]], device: Optional[Any] = None) -> None:
        for metric in self.metrics.values():
            metric.update(preds, targets, verbs, choices, device=device)

        # if ground_truth_prob_metric := self.metrics.get("ground_truth_prob"):
        #     assert label_prob is not None
        #     output["ground_truth_prob"] = ground_truth_prob_metric(label_prob)
        #
        # if perplexity_metric := self.metrics.get("perplexity"):
        #     assert label_probs is not None and perplexity_mask is not None
        #     output["perplexity"] = perplexity_metric(label_probs, perplexity_mask)

    @overrides
    def compute(self) -> Mapping[str, torch.Tensor]:
        return {name: metrics.compute() for name, metrics in self.metrics.items()}
