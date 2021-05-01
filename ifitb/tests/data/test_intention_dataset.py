from unittest import TestCase

from ifitb.data.data_module import URL_INTENTIONS_DATA, URL_REASONS_BY_VERB
from ifitb.data.intention_dataset import IntentionDataset


class TestIntentionDataset(TestCase):
    def test_dataset_format(self):
        dataset = IntentionDataset(URL_REASONS_BY_VERB, URL_INTENTIONS_DATA)
        expected_first_item = {
            "text": "I had to wake up early for that And I had to get ready put on makeup and start a eventful and epic"
                    " day It was a very fulfilling day but I was exhausted because I got like four hours of sleep as"
                    " did all my friends who slept over but it was a super fun day where we got to celebrate because"
                    " _____ the launch of the Artist of Life The time is 12 00 12 02 Workbook and the Daily Planner"
                    " and I got to meet a lot of you face to face for the first time so that was a real treat",
            "video_id": "3LJdjNMezJc",
            "video_end_time": "0:08:34.080000",
            "video_start_time": "0:07:58.900000",
            "choices": ["mark happy occasion", "acknowledge accomplishment", "have good time", "share happiness"],
        }
        actual_first_item = dataset[0]
        self.assertEqual(expected_first_item, actual_first_item)
