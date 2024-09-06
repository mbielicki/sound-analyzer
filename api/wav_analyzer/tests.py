from django.test import TestCase

from api.wav_analyzer.analyze import fs_to_notes, wav_to_fs
from api.wav_analyzer.notes import f_to_note, note_n_to_name, note_name_to_n
from api.wav_analyzer.wav import make_waves, to_wav_bytes

class FftTestCase(TestCase):
    def test_detect_waves(self):
        freqs = [
            ('A0', 0.3),
            ('A3', 0.2),
            ('C4', 0.1),
            ('E4', 0.1),
            ('A6', 0.05)
        ]
        chunk = to_wav_bytes(make_waves(freqs, duration=0.5)[1])
        xf, yf = wav_to_fs(chunk)
        notes = fs_to_notes(xf, yf)
        self.assertIn(note_name_to_n('C4'), notes)


class NoteConversionTestCase(TestCase):
    def test_f_to_note(self):
        self.assertEqual(f_to_note(28), 0)
        self.assertEqual(f_to_note(440), 48)

        self.assertEqual(f_to_note(27.5), 0)
        self.assertEqual(f_to_note(4186.009), 87)

    def test_out_of_bounds_f_to_note(self):
        self.assertEqual(f_to_note(4434.922), None)
        self.assertEqual(f_to_note(25.9565), None)
        self.assertEqual(f_to_note(1e-10000), None)

    def test_n_to_name(self):
        self.assertEqual(note_n_to_name(0), 'A0')
        self.assertEqual(note_n_to_name(1), 'A#0')
        self.assertEqual(note_n_to_name(2), 'B0')
        self.assertEqual(note_n_to_name(3), 'C1')
        self.assertEqual(note_n_to_name(15), 'C2')
        self.assertEqual(note_n_to_name(87), 'C8')

    def test_error_n_to_name(self):
        self.assertRaises(ValueError, note_n_to_name, -1)
        self.assertRaises(ValueError, note_n_to_name, 88)

    def test_name_to_n(self):
        self.assertEqual(note_name_to_n('A0'), 0)
        self.assertEqual(note_name_to_n('A#0'), 1)
        self.assertEqual(note_name_to_n('B0'), 2)
        self.assertEqual(note_name_to_n('C1'), 3)
        self.assertEqual(note_name_to_n('C8'), 87)

    def test_error_name_to_n(self):
        self.assertRaises(ValueError, note_name_to_n, 'what')
        self.assertRaises(ValueError, note_name_to_n, 'X4')
        self.assertRaises(ValueError, note_name_to_n, 'E#4')
        self.assertRaises(ValueError, note_name_to_n, 'C#8')
        self.assertRaises(ValueError, note_name_to_n, 'C#10')
        self.assertRaises(ValueError, note_name_to_n, 'G#0')