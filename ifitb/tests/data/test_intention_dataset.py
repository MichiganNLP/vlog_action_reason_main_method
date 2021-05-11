from unittest import TestCase

from ifitb.data.data_module import URL_INTENTIONS_TEST
from ifitb.data.intention_dataset import IntentionDataset


class TestIntentionDataset(TestCase):
    def test_dataset_format(self):
        dataset = IntentionDataset(URL_INTENTIONS_TEST)
        expected_first_item = {
            "text": "hey everyone welcome back to my channel so as promised i said i would do some weekend reset"
                    " videos my last one was pretty much completely cleaning because _____ because that weekend i"
                    " really needed to clean and organize to get everything ready for the new week ahead as well and i"
                    " had to do that this past weekend",
            "video_id": "EphWUUqxbck",
            "video_start_time": "0:00:04.400000",
            "video_end_time": "0:00:27.119000",
            "verb": "clean",
            "choices": ["company was coming", "do not like dirtiness", "habit", "self care", "declutter", "remove dirt",
                        "I cannot find any reason mentioned verbally or shown visually in the video"],
            "ground_truth": ["declutter", "remove dirt"],
        }
        actual_first_item = dataset[0]
        self.assertEqual(expected_first_item, actual_first_item)
