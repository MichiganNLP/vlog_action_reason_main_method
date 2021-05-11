from unittest import TestCase

from ifitb.metrics.metrics import AllMetrics


class TestAllMetrics(TestCase):
    def test_all_metrics(self):
        metric = AllMetrics()

        metric.update([[1.0, 0.0, 0.0], [1.0, 1.0]],
                      [["guests are coming"], ["fitness"]],
                      ["clean", "jump"],
                      [["dirty", "guests are coming", "tidy"], ["is fun!", "fitness"]])

        metric.update([[0.0, 1.0, 1.0]],
                      [["guests are coming", "dirty"]],
                      ["clean"],
                      [["dirty", "guests are coming", "tidy"]])

        expected = {
            "accuracy": ((1 / 3 + 1 / 3) / 2 + 1 / 2) / 2,
            "precision": ((0 + 1 / 2) / 2 + 1 / 2) / 2,
            "recall": ((0 + 1 / 2) / 2 + 1) / 2,
            "f1": ((0 + 1 / (1 + (1 + 1) / 2)) / 2 + 1 / (1 + 1 / 2)) / 2,
        }
        actual = metric.compute()

        self.assertEqual(set(expected), set(actual))

        for k in expected:
            self.assertAlmostEqual(expected[k], actual[k].item(), msg=k)
