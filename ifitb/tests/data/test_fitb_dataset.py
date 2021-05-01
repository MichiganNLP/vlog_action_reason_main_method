from unittest import TestCase

from ifitb.data.fitb_dataset import _blank_reason, _format_blanks_for_t5


class TestFitbDataset(TestCase):
    def test_blank_reason(self):
        text_with_blank, label = _blank_reason("I dance because I love dancing")
        self.assertEqual("I dance because _____", text_with_blank)
        self.assertEqual("I love dancing", label)

    def test_blank_reason_empty(self):
        text_with_blank, label = _blank_reason("It's not blowing and making bottles")
        self.assertEqual("It's not blowing and making bottles", text_with_blank)
        self.assertEqual(None, label)

    def test_format_blanks_for_t5(self):
        text_with_blanks = _format_blanks_for_t5("I love how _____ brings _____")
        self.assertEqual("I love how <extra_id_0> brings <extra_id_1>", text_with_blanks)
