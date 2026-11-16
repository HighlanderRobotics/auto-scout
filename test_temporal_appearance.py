import unittest
from train_temporal_appearance import exact_identity, split_index


class TemporalAppearanceTests(unittest.TestCase):
    def test_conflicting_ocr_or_partial_number_never_names_a_crop(self):
        roster={'3197','2194'}
        self.assertEqual(exact_identity([dict(team='3197',confidence=.9)],roster),'3197')
        self.assertIsNone(exact_identity([dict(team='3197',confidence=.9),dict(team='2194',confidence=.9)],roster))
        self.assertIsNone(exact_identity([dict(team='319',confidence=1.)],roster))
        self.assertIsNone(exact_identity([dict(team='3197',confidence=.7)],roster))

    def test_holdout_has_two_second_gap_and_no_overlap(self):
        self.assertEqual(split_index(59),'train')
        self.assertTrue(all(split_index(i) is None for i in range(60,70)))
        self.assertEqual(split_index(70),'test')


if __name__=='__main__':unittest.main()
