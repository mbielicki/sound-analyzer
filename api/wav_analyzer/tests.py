from django.test import TestCase
from ..wav_analyzer.utils import f_to_key

class FToKeyTestCase(TestCase):
    def test_f_to_key(self):
        self.assertEqual(f_to_key(28), 0)
        self.assertEqual(f_to_key(440), 48)
