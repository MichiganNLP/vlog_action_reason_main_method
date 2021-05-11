from unittest import TestCase

from ifitb.metrics.accuracy import AccuracyPerAction


class TestAccuracyPerAction(TestCase):
    def test_accuracy_per_action(self):
        metric = AccuracyPerAction()

        metric.update([["dirty"], ["fitness", "is fun!"]],
                      [["guests are coming"], ["fitness"]],
                      ["clean", "jump"],
                      [["dirty", "guests are coming", "tidy"], ["is fun!", "fitness"]])

        metric.update([["guests are coming", "tidy"]],
                      [["guests are coming", "dirty"]],
                      ["clean"],
                      [["dirty", "guests are coming", "tidy"]])

        self.assertAlmostEqual((2 / 6 + 1 / 2) / 2, metric.compute().item())